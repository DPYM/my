import torch
import torch.nn as nn
import math
import config

#多头注意力机制
class Multihead_attention(nn.Module):
    def __init__(self,dim,n_heads,drop_out):
        super().__init__()
        self.dim=dim
        self.n_heads=n_heads
        self.head_dim=dim//n_heads  #每个头的维度，也就是根号下dk

        #QKV的线性变换
        self.linear_Q=nn.Linear(dim,dim)
        self.linear_K=nn.Linear(dim,dim)
        self.linear_V=nn.Linear(dim,dim)

        self.output=nn.Linear(dim,dim)
        self.drop_out=nn.Dropout(drop_out)  #添加dropout正则化
        self.layer_norm=nn.LayerNorm(dim)

        #添加FNN前馈神经网络
        self.fnn=nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(dim*4,dim),
            nn.Dropout(drop_out)
        )
        self.fnn_norm=nn.LayerNorm(dim)

    def forward(self,query,key,value,key_padding_mask=None):
        batch_size=query.size(0)
        q_seq_len=query.size(1)
        k_seq_len=key.size(1)
        v_seq_len=value.size(1)

        #线性变换
        Q=self.linear_Q(query)
        K=self.linear_K(key)
        V=self.linear_V(value)

        Q=Q.view(batch_size,q_seq_len,self.n_heads,self.head_dim).transpose(1,2)
        K=K.view(batch_size,k_seq_len,self.n_heads,self.head_dim).transpose(1,2)
        V=V.view(batch_size,v_seq_len,self.n_heads,self.head_dim).transpose(1,2)

        scores=torch.matmul(Q,K.transpose(2,3))/(self.head_dim**0.5)#Q与K的转置的点积
        if key_padding_mask is not None:
            mask=key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores=scores.masked_fill(mask,torch.finfo(scores.dtype).min)
        attention_weights=torch.softmax(scores,dim=-1)
        attention_weights=self.drop_out(attention_weights)#增加一个dropout层
        attention=torch.matmul(attention_weights,V)#每个物品分配到的注意力权重
        attention=attention.transpose(1,2).contiguous().view(batch_size,q_seq_len,self.dim)

        #残差链接，防止梯度消失
        attention=self.drop_out(self.output(attention))
        attention=self.layer_norm(query+attention)

        #FNN层，GELU激活
        fnn_out=self.fnn(attention)
        output=self.fnn_norm(attention+fnn_out)
        
        return output

#引入位置编码，捕捉行为序列的时间顺序信息
class PositionalEncoding(nn.Module):
    def __init__(self,dim,max_length=500):
        super().__init__()
        pe=torch.zeros(max_length,dim)
        position=torch.arange(0,max_length,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,dim,2).float()*(-math.log(10000)/dim))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        return x+self.pe[:,:x.size(1),:]

class Multihead_interest(nn.Module):
    def __init__(self,n_users,n_movies,dim,n_interests,n_heads,n_tags=0,drop_out=config.multihead_dropout):
        super(Multihead_interest,self).__init__()
        self.user_embedding=nn.Embedding(n_users,dim)
        self.movie_embedding=nn.Embedding(n_movies,dim,padding_idx=0)
        self.n_interests=n_interests #k个兴趣向量
        self.dim=dim #维度
        self.n_tags=n_tags
        self.multihead_attention=Multihead_attention(dim=dim,n_heads=n_heads,drop_out=drop_out)
        
        if n_tags>0:
            self.tag_embedding=nn.Linear(n_tags,dim,bias=False)



        self.position_encoding=PositionalEncoding(dim)

        #使用xavier初始化
        interest_vecs=torch.empty(n_interests,dim)
        nn.init.xavier_uniform_(interest_vecs)
        self.interest_vecs=nn.Parameter(interest_vecs)

        #兴趣门控机制，学习每个兴趣的重要性
        self.interest_gate=nn.Sequential(
            nn.Linear(dim*2,dim),
            nn.Sigmoid()
        )

        #兴趣融合层
        self.interest_fusion=nn.Sequential(
            nn.Linear(dim*2,dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(drop_out),
            nn.Linear(dim,dim)
        )
        
        #投影层
        self.output_projection=nn.Linear(dim,dim)
        
        #评分函数
        self.scorer=nn.Sequential(
            nn.Linear(dim*2,dim),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(dim,1)
        )

    def get_user_representation(self,user_ids,user_history,history_tags=None):
        user_vecs=self.user_embedding(user_ids)
        history_embedding=self.movie_embedding(user_history)
        
        if history_tags is not None and self.n_tags>0:
            tag_emb=self.tag_embedding(history_tags)
            history_embedding=history_embedding+tag_emb
        
        user_expanded=user_vecs.unsqueeze(1).expand(-1,self.n_interests,-1)

        history_embedding=self.position_encoding(history_embedding)

        padding_mask=(user_history)==0

        interest_vecs=self.interest_vecs.unsqueeze(0)
        batch=user_vecs.shape[0]
        interest_query=interest_vecs.expand(batch,-1,-1)+user_vecs.unsqueeze(1)

        attention_out=self.multihead_attention(
            query=interest_query,
            key=history_embedding,
            value=history_embedding,
            key_padding_mask=padding_mask
        )

        gate_input=torch.cat([user_expanded,attention_out],dim=-1)
        gate_weights=self.interest_gate(gate_input)
        gated_interest=attention_out*gate_weights

        fused_interest=torch.cat([user_expanded,gated_interest],dim=-1)
        final_interest=self.interest_fusion(fused_interest)

        final_interest=self.output_projection(final_interest)

        return final_interest

    def forward(self,user_ids,user_history,history_tags,movie_ids):
        user_repr=self.get_user_representation(user_ids,user_history,history_tags)
        movie_emb=self.movie_embedding(movie_ids)
        
        if movie_emb.dim()==2:
            movie_emb=movie_emb.unsqueeze(1)
        
        batch,n_cand,_=movie_emb.shape
        user_expanded=user_repr.unsqueeze(2).expand(-1,-1,n_cand,-1)
        movie_expanded=movie_emb.unsqueeze(1).expand(-1,self.n_interests,-1,-1)
        
        #拼接特征
        combined=torch.cat([user_expanded,movie_expanded],dim=-1)
        
        #评分
        scores=self.scorer(combined).squeeze(-1)
        scores=scores.max(dim=1)[0]
        
        if n_cand==1:
            scores=scores.squeeze(1)
        
        return scores