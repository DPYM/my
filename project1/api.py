import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from inference.pipeline import Pipeline
from inference.ab_test import ABTestManager
import uvicorn

app=FastAPI(title='Movie Recommender API',version='2.0')
pipeline=None
ab_manager=None

EXPERIMENT_NAME='mmr_pop_vs_deepfm'
EXPERIMENT_CONFIG={
    'description': 'MMR+Pop vs 仅DeepFM',
    'groups': {
        'control': {
            'name': '仅DeepFM',
            'description': '热门候选→DeepFM精排',
            'ratio': 0.5,
        },
        'treatment': {
            'name': 'MMR+Pop',
            'description': 'MIND召回→DeepFM精排→MMR多样性重排+流行度惩罚',
            'ratio': 0.5,
        }
    }
}

class RecommendRequest(BaseModel):
    user_id:int
    top_n:Optional[int]=20
    use_cache:Optional[bool]=True
    experiment:Optional[str]=None
    force_group:Optional[str]=None

class MovieItem(BaseModel):
    movie_id:int
    title:str
    movie_type:str
    score:float
    reason:Optional[str]=None

class RecommendResponse(BaseModel):
    user_id:int
    group:str
    recommendations:List[MovieItem]

class BatchRequest(BaseModel):
    user_ids:List[int]
    top_n:Optional[int]=20

class SimilarMovieRequest(BaseModel):
    movie_name:str
    top_n:Optional[int]=10

class SimilarMovieItem(BaseModel):
    movie_id:int
    title:str
    genres:str
    similarity:float

class MovieSearchResponse(BaseModel):
    input_movie:dict
    recommendations:List[SimilarMovieItem]

@app.on_event('startup')
def load_pipeline():
    global pipeline, ab_manager
    print('Loading recommendation pipeline...')
    pipeline=Pipeline(recall_top_k=800,coarse_rank_top=400,fine_rank_top=20)
    print('Pipeline loaded successfully')

    ab_manager=ABTestManager()
    ab_manager.add_experiment(EXPERIMENT_NAME, EXPERIMENT_CONFIG)
    print(f'A/B Test initialized: {EXPERIMENT_NAME}')
    print(f'  Control: 仅DeepFM (50%)')
    print(f'  Treatment: MMR+Pop (50%)')

@app.get('/health')
def health_check():
    return {'status':'ok','ab_test':EXPERIMENT_NAME}

@app.post('/recommend',response_model=RecommendResponse)
def recommend(req:RecommendRequest):
    if pipeline is None:
        raise HTTPException(status_code=503,detail='Pipeline not loaded')

    experiment=req.experiment or EXPERIMENT_NAME
    if req.force_group and req.force_group in ('control','treatment'):
        group=req.force_group
    else:
        group=ab_manager.get_group(req.user_id, experiment)

    start=time.time()
    try:
        if group=='treatment':
            result=pipeline.recommend_mmr_pop(user_id=req.user_id,top_n=req.top_n,lam=0.6,pop_weight=0.2)
        else:
            result=pipeline.recommend_deepfm_only(user_id=req.user_id,top_n=req.top_n)

        latency_ms=(time.time()-start)*1000
        ab_manager.log_recommendation(experiment, group, req.user_id, result, latency_ms)

        return RecommendResponse(
            user_id=req.user_id,
            group=group,
            recommendations=[MovieItem(**r) for r in result]
        )
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.post('/batch_recommend')
def batch_recommend(req:BatchRequest):
    if pipeline is None:
        raise HTTPException(status_code=503,detail='Pipeline not loaded')
    try:
        result={}
        for uid in req.user_ids:
            group=ab_manager.get_group(uid, EXPERIMENT_NAME)
            start=time.time()
            if group=='treatment':
                rec=pipeline.recommend_mmr_pop(user_id=uid,top_n=req.top_n,lam=0.6,pop_weight=0.2)
            else:
                rec=pipeline.recommend_deepfm_only(user_id=uid,top_n=req.top_n)
            latency_ms=(time.time()-start)*1000
            ab_manager.log_recommendation(EXPERIMENT_NAME, group, uid, rec, latency_ms)
            result[uid]={'group':group,'recommendations':rec}
        return result
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.get('/ab/metrics')
def get_ab_metrics(experiment:Optional[str]=None):
    if ab_manager is None:
        raise HTTPException(status_code=503,detail='A/B manager not initialized')
    exp=experiment or EXPERIMENT_NAME
    return ab_manager.get_metrics(exp)

@app.get('/ab/compare')
def compare_ab(experiment:Optional[str]=None):
    if ab_manager is None:
        raise HTTPException(status_code=503,detail='A/B manager not initialized')
    exp=experiment or EXPERIMENT_NAME
    return ab_manager.compare(exp)

@app.get('/ab/logs')
def get_ab_logs(experiment:Optional[str]=None, group:Optional[str]=None, limit:int=50):
    if ab_manager is None:
        raise HTTPException(status_code=503,detail='A/B manager not initialized')
    exp=experiment or EXPERIMENT_NAME
    return ab_manager.get_logs(exp, group, limit)

@app.post('/ab/reset')
def reset_ab(experiment:Optional[str]=None):
    if ab_manager is None:
        raise HTTPException(status_code=503,detail='A/B manager not initialized')
    ab_manager.reset(experiment)
    return {'status':'ok','message':'A/B test data reset'}

@app.get('/ab/config')
def get_ab_config():
    return {
        'experiment': EXPERIMENT_NAME,
        'config': EXPERIMENT_CONFIG,
    }

@app.post('/similar_movies',response_model=MovieSearchResponse)
def similar_movies(req:SimilarMovieRequest):
    if pipeline is None:
        raise HTTPException(status_code=503,detail='Pipeline not loaded')
    try:
        movie_name=req.movie_name
        translated=pipeline.translator.translate(movie_name)
        if translated:
            movie_name=translated
        result=pipeline.recommend_by_movie(movie_name=movie_name,top_n=req.top_n)
        if 'error' in result:
            raise HTTPException(status_code=404,detail=result['error'])
        return MovieSearchResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

if __name__=='__main__':
    uvicorn.run(app,host='0.0.0.0',port=8000)
