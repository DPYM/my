import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
import config
from inference.Rank import Rank
from inference.CoarseRank import CoarseRank
from inference.recall import Recall
from inference.translator import MovieTranslator
import time

class Pipeline:
    def __init__(self,recall_top_k=500,coarse_rank_top=200,fine_rank_top=20,user_rerank=True):
        self.recall_top_k=recall_top_k
        self.coarse_rank_top_k=coarse_rank_top
        self.fine_rank_top_k=fine_rank_top
        self.user_rerank=user_rerank

        with open(config.user_history_path,'rb') as f:
            self.user_history=pickle.load(f)
        with open(config.movie_encoder_path,'rb') as f:
            self.movie_encoder=pickle.load(f)
        self.movie_data = pd.read_csv(config.movies_path)
        self._ratings_data = None
        self._movie_counts_cache = None
        self._max_movie_count = None
        self.original_to_encoded={}
        for i,orig_id in enumerate(self.movie_encoder.classes_):
            self.original_to_encoded[int(orig_id)]=i+1
        self.encoded_to_original={}
        for orig_id,enc_id in self.original_to_encoded.items():
            self.encoded_to_original[enc_id]=orig_id

        self.recall_engine=Recall()
        self.recall_engine.build_index()
        self.coarse_ranker=CoarseRank()
        self.fine_ranker=Rank()
        self.cache={}
        self.cache_ttl={}
        self.cache_ttl_second=1800
        
        self.translator = MovieTranslator()
        self._popular_movies = None

    @property
    def ratings_data(self):
        if self._ratings_data is None:
            self._ratings_data = pd.read_csv(config.ratings_path, usecols=['movieId'])
        return self._ratings_data

    def recommend_deepfm_only(self, user_id, top_n=20):
        movie_seen=set(self.user_history.get(user_id,[]))
        popular_original=self._get_popular_movies(top_n*20)
        candidates=[]
        for orig_mid in popular_original:
            enc_mid=self.original_to_encoded.get(orig_mid)
            if enc_mid is not None and enc_mid not in movie_seen:
                candidates.append(enc_mid)
            if len(candidates)>=top_n*3:
                break
        if not candidates:
            return self.fallback_hot(user_id, top_n)
        ranked=self.fine_rank(user_id,candidates,top_n*3)
        result=self.format_output(ranked[:top_n])
        return result

    def recommend_no_coarse(self,user_id,top_n=20):
        movie_seen=set(self.user_history.get(user_id,[]))
        recall_list=self.recall(user_id)
        if not recall_list:
            return self.fallback_hot(user_id, top_n)
        candidates=[mid for mid in recall_list if mid not in movie_seen]
        if not candidates:
            return self.fallback_hot(user_id, top_n)
        fine_ranked=self.fine_rank(user_id,candidates,top_n*10)
        if self.user_rerank:
            reranked=self.rerank(user_id,fine_ranked,top_n)
        else:
            reranked=fine_ranked[:top_n]
        result=self.format_output(reranked[:top_n])
        return result

    def recommend_blended(self,user_id,top_n=20,alpha=0.5):
        movie_seen=set(self.user_history.get(user_id,[]))
        recall_list,recall_scores=self.recall(user_id,return_scores=True)
        if not recall_list:
            return self.fallback_hot(user_id, top_n)

        recall_score_map={}
        for mid,score in zip(recall_list,recall_scores):
            if mid not in movie_seen:
                recall_score_map[mid]=score

        if not recall_score_map:
            return self.fallback_hot(user_id, top_n)

        candidates=list(recall_score_map.keys())
        fine_ranked=self.fine_rank(user_id,candidates,len(candidates))

        deepfm_score_map={}
        for mid,score in fine_ranked:
            deepfm_score_map[mid]=score

        max_recall=max(recall_score_map.values()) if recall_score_map else 1.0
        min_recall=min(recall_score_map.values()) if recall_score_map else 0.0
        recall_range=max_recall-min_recall if max_recall>min_recall else 1.0

        deepfm_scores=list(deepfm_score_map.values())
        max_deepfm=max(deepfm_scores) if deepfm_scores else 1.0
        min_deepfm=min(deepfm_scores) if deepfm_scores else 0.0
        deepfm_range=max_deepfm-min_deepfm if max_deepfm>min_deepfm else 1.0

        blended=[]
        for mid in candidates:
            norm_recall=(recall_score_map[mid]-min_recall)/recall_range
            norm_deepfm=(deepfm_score_map.get(mid,0)-min_deepfm)/deepfm_range
            final_score=alpha*norm_recall+(1-alpha)*norm_deepfm
            blended.append((mid,final_score))

        blended.sort(key=lambda x:x[1],reverse=True)

        if self.user_rerank:
            reranked=self.rerank(user_id,blended,top_n)
        else:
            reranked=blended[:top_n]
        result=self.format_output(reranked[:top_n])
        return result

    def _mmr_select(self,rel_map,candidates,top_n,lam):
        if not hasattr(self,'_movie_emb_matrix'):
            self._build_movie_emb_matrix()

        emb=self._movie_emb_matrix
        cands=list(candidates)
        valid_cands=[m for m in cands if m<len(emb)]
        vecs=emb[valid_cands]
        norms=np.linalg.norm(vecs,axis=1,keepdims=True)
        norms[norms==0]=1.0
        norm_vecs=vecs/norms

        selected=[]
        remaining_idx=set(range(len(valid_cands)))
        mid_to_idx={m:i for i,m in enumerate(valid_cands)}

        best_i=max(remaining_idx,key=lambda i:rel_map[valid_cands[i]])
        selected.append(valid_cands[best_i])
        remaining_idx.remove(best_i)

        for _ in range(min(top_n-1,len(valid_cands)-1)):
            if not remaining_idx:
                break
            sel_vecs=norm_vecs[[mid_to_idx[m] for m in selected]]
            best_i=None
            best_score=-float('inf')
            for i in remaining_idx:
                relevance=rel_map[valid_cands[i]]
                sims=sel_vecs@norm_vecs[i]
                max_sim=sims.max()
                mmr_score=lam*relevance-(1-lam)*max_sim
                if mmr_score>best_score:
                    best_score=mmr_score
                    best_i=i
            if best_i is not None:
                selected.append(valid_cands[best_i])
                remaining_idx.remove(best_i)

        return selected

    def recommend_mmr(self,user_id,top_n=20,lam=0.6):
        movie_seen=set(self.user_history.get(user_id,[]))
        recall_list,recall_scores=self.recall(user_id,return_scores=True)
        if not recall_list:
            return self.fallback_hot(user_id, top_n)

        recall_score_map={}
        for mid,score in zip(recall_list,recall_scores):
            if mid not in movie_seen:
                recall_score_map[mid]=score

        if not recall_score_map:
            return self.fallback_hot(user_id, top_n)

        top_cands=sorted(recall_score_map.keys(),key=lambda m:recall_score_map[m],reverse=True)[:200]
        fine_ranked=self.fine_rank(user_id,top_cands,200)

        deepfm_score_map={}
        for mid,score in fine_ranked:
            deepfm_score_map[mid]=score

        recall_vals=[recall_score_map[m] for m in top_cands]
        deepfm_vals=[deepfm_score_map.get(m,0) for m in top_cands]
        max_r,min_r=max(recall_vals),min(recall_vals)
        max_d,min_d=max(deepfm_vals),min(deepfm_vals)
        r_range=max_r-min_r if max_r>min_r else 1.0
        d_range=max_d-min_d if max_d>min_d else 1.0

        rel_map={}
        for mid in top_cands:
            norm_r=(recall_score_map[mid]-min_r)/r_range
            norm_d=(deepfm_score_map.get(mid,0)-min_d)/d_range
            rel_map[mid]=0.5*norm_r+0.5*norm_d

        selected=self._mmr_select(rel_map,top_cands,top_n,lam)
        result_items=[(mid,rel_map[mid]) for mid in selected]
        result=self.format_output(result_items[:top_n])
        return result

    def _build_movie_emb_matrix(self):
        movie_emb=self.fine_ranker.model.movie_embedding.weight.detach().cpu().numpy()
        self._movie_emb_matrix=movie_emb

    def recommend_mmr_pop(self,user_id,top_n=20,lam=0.6,pop_weight=0.3):
        movie_seen=set(self.user_history.get(user_id,[]))
        recall_list,recall_scores=self.recall(user_id,return_scores=True)
        if not recall_list:
            return self.fallback_hot(user_id, top_n)

        recall_score_map={}
        for mid,score in zip(recall_list,recall_scores):
            if mid not in movie_seen:
                recall_score_map[mid]=score

        if not recall_score_map:
            return self.fallback_hot(user_id, top_n)

        top_cands=sorted(recall_score_map.keys(),key=lambda m:recall_score_map[m],reverse=True)[:200]
        fine_ranked=self.fine_rank(user_id,top_cands,200)

        deepfm_score_map={}
        for mid,score in fine_ranked:
            deepfm_score_map[mid]=score

        recall_vals=[recall_score_map[m] for m in top_cands]
        deepfm_vals=[deepfm_score_map.get(m,0) for m in top_cands]
        max_r,min_r=max(recall_vals),min(recall_vals)
        max_d,min_d=max(deepfm_vals),min(deepfm_vals)
        r_range=max_r-min_r if max_r>min_r else 1.0
        d_range=max_d-min_d if max_d>min_d else 1.0

        if not hasattr(self,'_movie_counts_cache'):
            self._movie_counts_cache=dict(self.ratings_data['movieId'].value_counts())
            self._max_movie_count=max(self._movie_counts_cache.values()) if self._movie_counts_cache else 1

        rel_map={}
        for mid in top_cands:
            norm_r=(recall_score_map[mid]-min_r)/r_range
            norm_d=(deepfm_score_map.get(mid,0)-min_d)/d_range
            base_rel=0.5*norm_r+0.5*norm_d
            orig_id=self.encoded_to_original.get(mid,mid)
            count=self._movie_counts_cache.get(orig_id,0)
            popularity=count/self._max_movie_count
            rel_map[mid]=base_rel*(1-pop_weight*popularity)

        selected=self._mmr_select(rel_map,top_cands,top_n,lam)
        result_items=[(mid,rel_map[mid]) for mid in selected]
        result=self.format_output(result_items[:top_n])
        return result

    def recommend_rrf(self,user_id,top_n=20,k=60):
        movie_seen=set(self.user_history.get(user_id,[]))
        recall_list,recall_scores=self.recall(user_id,return_scores=True)
        if not recall_list:
            return self.fallback_hot(user_id, top_n)

        recall_candidates=[]
        for mid,score in zip(recall_list,recall_scores):
            if mid not in movie_seen:
                recall_candidates.append((mid,score))
        if not recall_candidates:
            return self.fallback_hot(user_id, top_n)

        recall_candidates.sort(key=lambda x:x[1],reverse=True)
        recall_rank={mid:i+1 for i,(mid,_) in enumerate(recall_candidates)}

        candidates=[mid for mid,_ in recall_candidates]
        fine_ranked=self.fine_rank(user_id,candidates,len(candidates))

        deepfm_rank={mid:i+1 for i,(mid,_) in enumerate(fine_ranked)}

        rrf_scores={}
        for mid in candidates:
            r1=recall_rank.get(mid,len(recall_candidates)+1)
            r2=deepfm_rank.get(mid,len(candidates)+1)
            rrf_scores[mid]=1.0/(k+r1)+1.0/(k+r2)

        sorted_items=sorted(rrf_scores.items(),key=lambda x:x[1],reverse=True)

        if self.user_rerank:
            reranked=self.rerank(user_id,sorted_items,top_n)
        else:
            reranked=sorted_items[:top_n]
        result=self.format_output(reranked[:top_n])
        return result

    def recommend_pop_penalty(self,user_id,top_n=20,beta=0.2):
        movie_seen=set(self.user_history.get(user_id,[]))
        recall_list,recall_scores=self.recall(user_id,return_scores=True)
        if not recall_list:
            return self.fallback_hot(user_id, top_n)

        recall_score_map={}
        for mid,score in zip(recall_list,recall_scores):
            if mid not in movie_seen:
                recall_score_map[mid]=score

        if not recall_score_map:
            return self.fallback_hot(user_id, top_n)

        candidates=list(recall_score_map.keys())
        fine_ranked=self.fine_rank(user_id,candidates,len(candidates))

        if not hasattr(self,'_movie_counts_cache'):
            self._movie_counts_cache=dict(self.ratings_data['movieId'].value_counts())
            self._max_movie_count=max(self._movie_counts_cache.values()) if self._movie_counts_cache else 1

        penalized=[]
        for mid,score in fine_ranked:
            orig_id=self.encoded_to_original.get(mid,mid)
            count=self._movie_counts_cache.get(orig_id,0)
            popularity=count/self._max_movie_count
            penalized_score=score*(1-beta*popularity)
            penalized.append((mid,penalized_score))

        penalized.sort(key=lambda x:x[1],reverse=True)

        if self.user_rerank:
            reranked=self.rerank(user_id,penalized,top_n)
        else:
            reranked=penalized[:top_n]
        result=self.format_output(reranked[:top_n])
        return result

    def recommend_zscore_blended(self,user_id,top_n=20,alpha=0.5):
        movie_seen=set(self.user_history.get(user_id,[]))
        recall_list,recall_scores=self.recall(user_id,return_scores=True)
        if not recall_list:
            return self.fallback_hot(user_id, top_n)

        recall_score_map={}
        for mid,score in zip(recall_list,recall_scores):
            if mid not in movie_seen:
                recall_score_map[mid]=score

        if not recall_score_map:
            return self.fallback_hot(user_id, top_n)

        candidates=list(recall_score_map.keys())
        fine_ranked=self.fine_rank(user_id,candidates,len(candidates))

        deepfm_score_map={}
        for mid,score in fine_ranked:
            deepfm_score_map[mid]=score

        recall_vals=np.array(list(recall_score_map.values()))
        recall_mean=recall_vals.mean()
        recall_std=recall_vals.std() if recall_vals.std()>0 else 1.0

        deepfm_vals=np.array(list(deepfm_score_map.values()))
        deepfm_mean=deepfm_vals.mean()
        deepfm_std=deepfm_vals.std() if deepfm_vals.std()>0 else 1.0

        blended=[]
        for mid in candidates:
            z_recall=(recall_score_map[mid]-recall_mean)/recall_std
            z_deepfm=(deepfm_score_map.get(mid,0)-deepfm_mean)/deepfm_std
            final_score=alpha*z_recall+(1-alpha)*z_deepfm
            blended.append((mid,final_score))

        blended.sort(key=lambda x:x[1],reverse=True)

        if self.user_rerank:
            reranked=self.rerank(user_id,blended,top_n)
        else:
            reranked=blended[:top_n]
        result=self.format_output(reranked[:top_n])
        return result

    def _get_popular_movies(self,n=500):
        if self._popular_movies is not None:
            return self._popular_movies[:n]
        hot=self.ratings_data['movieId'].value_counts()
        self._popular_movies=[int(mid) for mid in hot.index]
        return self._popular_movies[:n]

    def recommend(self,user_id,use_cache=True,top_n=None):
        if top_n is None:
            top_n=self.fine_rank_top_k
        if use_cache and self.hit_cache(user_id):
            cached=self.get_cache(user_id)
            return cached[:top_n]

        recall_list=self.recall(user_id)
        if not recall_list:
            return self.fallback_hot(user_id, top_n)

        coarse_ranked=self.coarse_rank(user_id,recall_list,self.coarse_rank_top_k)
        if not coarse_ranked:
            return self.fallback_hot(user_id, top_n)

        fine_ranked=self.fine_rank(user_id,[mid for mid,_ in coarse_ranked],top_n*10)

        if self.user_rerank:
            reranked=self.rerank(user_id,fine_ranked,top_n)
        else:
            reranked=fine_ranked[:top_n]

        result=self.format_output(reranked[:top_n])
        self.set_cache(user_id,result)
        return result

    def bat_recommend(self,user_ids,top_n=None,use_cache=True):
        top_n=top_n or self.fine_rank_top_k
        return {uid:self.recommend(uid,use_cache,top_n) for uid in user_ids}

    def search_movies(self,query,top_n=5):
        if not query or len(query)<2:
            return []
        
        search_query = query
        translated = self.translator.translate(query)
        if translated:
            search_query = translated
        
        query_lower=search_query.lower()
        matches=[]
        for _,row in self.movie_data.iterrows():
            title=str(row.get('title','')).lower()
            if query_lower in title:
                matches.append({
                    'movie_id':int(row['movieId']),
                    'title':row.get('title','未知'),
                    'genres':row.get('genres','未知')
                })
                if len(matches)>=top_n:
                    break
        return matches

    def recommend_by_movie(self,movie_name,top_n=10):
        matches=self.search_movies(movie_name,top_n=1)
        if not matches:
            return {'error':f'未找到电影: {movie_name}', 'suggestions': []}
        
        movie=matches[0]
        original_id=movie['movie_id']
        encoded_id=self.original_to_encoded.get(original_id)
        if encoded_id is None:
            return {'error':f'电影ID {original_id} 未在模型中找到', 'suggestions': []}
        
        movie_emb=self.recall_engine.movie_emb[encoded_id]
        movie_emb=movie_emb.reshape(1,-1).astype(float)
        
        top_k=top_n+20
        sims,idxs=self.recall_engine.search_index(movie_emb,top_k)
        
        results=[]
        for sim,idx in zip(sims,idxs):
            if idx==encoded_id:
                continue
            orig_idx=self.encoded_to_original.get(int(idx))
            if orig_idx is None:
                continue
            row=self.movie_data[self.movie_data['movieId']==orig_idx]
            if not row.empty:
                results.append({
                    'movie_id':int(orig_idx),
                    'title':row.iloc[0].get('title','未知'),
                    'genres':row.iloc[0].get('genres','未知'),
                    'similarity':round(float(sim),4)
                })
            if len(results)>=top_n:
                break
        
        return {
            'input_movie': movie,
            'recommendations': results
        }

    def warm_up(self):
        print('开始预热...')
        users = list(self.user_history.keys())
        if not users:
            print('无用户数据，预热跳过')
            return

        # 确保召回索引已构建
        self.recall_engine.ensure_index()
        print('  [1/3] Faiss 召回索引就绪')

        # 触发 CUDA kernel 编译和矩阵预计算
        dummy_user = users[0]
        dummy_hist = self.user_history[dummy_user]
        if len(dummy_hist) > config.max_history_length:
            dummy_hist = dummy_hist[:config.max_history_length]
        elif len(dummy_hist) < config.max_history_length:
            dummy_hist = list(dummy_hist) + [0] * (config.max_history_length - len(dummy_hist))

        import numpy as np
        dummy_hist_arr = np.array([dummy_hist], dtype=np.int64)
        self.recall_engine.recall(dummy_user, dummy_hist_arr, top_k=self.recall_top_k)
        print('  [2/4] MIND 召回 + Faiss 检索预热完成')

        # 粗排预热
        recall_list = self.recall(dummy_user)
        if recall_list:
            coarse_result = self.coarse_ranker.rank(
                dummy_user, recall_list, top_n=min(self.coarse_rank_top_k, len(recall_list)))
            print('  [3/4] Multihead 粗排预热完成')

            # 精排预热（触发 types/tags 矩阵构建 + CUDA 编译）
            coarse_ids = [mid for mid, _ in coarse_result]
            self.fine_ranker.rank(dummy_user, coarse_ids, top_n=min(self.fine_rank_top_k, len(coarse_ids)))
            print('  [4/4] DeepFM 精排预热完成')

        torch.cuda.empty_cache()
        print('预热完成')

    def recall(self,user_id,return_scores=False):
        engine=self.recall_engine
        history=self.user_history.get(user_id)
        if history is None or len(history)==0:
            return [] if not return_scores else ([],[])

        history_arr=np.array(history,dtype=np.int64)
        return engine.recall(user_id,history_arr,top_k=self.recall_top_k,return_scores=return_scores)

    def coarse_rank(self,user_id,candidates,top_n):
        movie_ids=[mid for mid in candidates[:self.recall_top_k]]
        return self.coarse_ranker.rank(user_id,movie_ids,top_n)

    def fine_rank(self,user_id,candidates,top_n):
        return self.fine_ranker.rank(user_id,candidates,top_n)

    def rerank(self, user_id, ranked_list, top_n):
        movie_seen = set(self.user_history.get(user_id, []))
        result = []
        consecutive_genre_count = 0
        last_types = set()

        # Pre-compute rank → penalty to avoid O(n²) scanning
        rank_penalty = {mid: 1.0 / (1.0 + 0.003 * i) for i, (mid, _) in enumerate(ranked_list)}

        for movie_id, score in ranked_list:
            if movie_id in movie_seen:
                continue
            orig_id = self.encoded_to_original.get(movie_id, movie_id)
            row = self.movie_data[self.movie_data['movieId'] == orig_id]
            current_type = set()
            if not row.empty:
                type_str = str(row.iloc[0].get('genres', ''))
                current_type = set(type_str.split('|'))
            if last_types and len(last_types & current_type) >= 3:
                consecutive_genre_count += 1
                if consecutive_genre_count >= 2:
                    continue
            else:
                consecutive_genre_count = 0

            final_score = score * rank_penalty.get(movie_id, 1.0)
            result.append((movie_id, final_score))
            last_types = current_type

            if len(result) >= top_n:
                break
        return result

    def fallback_hot(self, user_id, top_n):
        hot = self.ratings_data['movieId'].value_counts()
        movie_seen = set(self.user_history.get(user_id, []))
        orig_seen = set()
        for enc_mid in movie_seen:
            orig = self.encoded_to_original.get(enc_mid)
            if orig is not None:
                orig_seen.add(orig)

        result = []
        for movie_id in hot.index:
            if movie_id in orig_seen:
                continue
            row = self.movie_data[self.movie_data['movieId'] == movie_id]
            if not row.empty:
                title = row.iloc[0]['title']
                movie_type = row.iloc[0]['genres']
            else:
                title = '未知'
                movie_type = '未知'
            result.append({
                'movie_id': int(movie_id),
                'title': title,
                'movie_type': movie_type,
                'score': 0.0,
                'reason': '热门电影推荐'
            })
            if len(result) >= top_n:
                break
        return result

    def format_output(self,ranked):
        output=[]
        for movie_id,score in ranked:
            orig_id=self.encoded_to_original.get(movie_id,movie_id)
            row=self.movie_data[self.movie_data['movieId']==orig_id]
            if not row.empty:
                output.append({
                    'movie_id':int(orig_id),
                    'title':row.iloc[0].get('title','未知'),
                    'movie_type':row.iloc[0].get('genres','未知'),
                    'score':round(float(score),4)
                })
            else:
                output.append({
                    'movie_id':int(orig_id),
                    'title':'未知',
                    'movie_type':'未知',
                    'score':round(float(score),4)
                })

        return output

    def hit_cache(self,user_id):
        if user_id not in self.cache:
            return False
        if time.time()>self.cache_ttl.get(user_id,0):
            del self.cache[user_id]
            del self.cache_ttl[user_id]
            return False
        return True

    def get_cache(self,user_id):
        return self.cache.get(user_id,[])

    def set_cache(self,user_id,result):
        self.cache[user_id]=result
        self.cache_ttl[user_id]=time.time()+self.cache_ttl_second

if __name__=='__main__':
    print('初始化')
    pipeline=Pipeline()
    pipeline.warm_up()

    print('推荐结果：')
    result=pipeline.recommend(user_id=0,top_n=20)
    for i,r in enumerate(result,1):
        print(f"{i}:[{r['movie_id']}{r['title']}{r['movie_type']}{r['score']}]")
