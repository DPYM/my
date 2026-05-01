import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self,dim,max_len=500):
        super().__init__()
        pe=torch.zeros(max_len,dim)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,dim,2).float()/dim * (-math.log(10000.0)))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        return x+self.pe[:,:x.size(1),:]

class Mind(nn.Module):
    def __init__(self,n_users,n_movies,dim,n_interest,route,dropout=0.1):
        super().__init__()
        self.user_embedding=nn.Embedding(n_users,dim)
        self.movie_embedding=nn.Embedding(n_movies,dim,padding_idx=0)

        self.n_interests=n_interest
        self.dim=dim
        self.route=route

        self.position_encoding=PositionalEncoding(dim)

        self.primary_caps=nn.Sequential(
            nn.Linear(dim,dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.routing_t=nn.Parameter(torch.tensor(0.1))

        self.interest_refine_weight=nn.Parameter(torch.randn(n_interest,dim,dim))
        self.interest_refine_bias=nn.Parameter(torch.zeros(n_interest,dim))
        nn.init.xavier_uniform_(self.interest_refine_weight)

        self.user_gate=nn.Sequential(
            nn.Linear(dim,dim),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def squash(self,x):
        squared_norm=(x**2).sum(dim=-1,keepdim=True)
        scale=squared_norm/(1+squared_norm)
        return x*scale/torch.sqrt(squared_norm+(1e-7))

    def dynamic_routing(self,primary_caps):
        batch_size,seq_length,dim=primary_caps.shape

        b=torch.zeros(batch_size,seq_length,self.n_interests,device=primary_caps.device)
        for i in range(self.route):
            c=torch.softmax(b/self.routing_t,dim=2)

            s=torch.bmm(primary_caps.transpose(1,2),c).transpose(1,2)
            v=self.squash(s)

            if i<self.route-1:
                sim=torch.bmm(primary_caps,v.transpose(1,2))
                sim=sim/(dim**0.5)
                b=b+sim
                b=torch.clamp(b,-10,10)

        return v

    def forward(self,user_ids,user_history):
        user_vecs=self.user_embedding(user_ids)
        history_vecs_clamped=torch.clamp(user_history,0,self.movie_embedding.num_embeddings-1)
        history_vecs=self.movie_embedding(history_vecs_clamped)

        history_vecs=self.position_encoding(history_vecs)

        primary_caps=self.squash(self.primary_caps(history_vecs))

        interest_vecs=self.dynamic_routing(primary_caps)

        B=interest_vecs.shape[0]
        weight=self.interest_refine_weight.unsqueeze(0).expand(B,-1,-1,-1)
        refined_interests=torch.matmul(interest_vecs.unsqueeze(2),weight).squeeze(2)+self.interest_refine_bias

        gate=self.user_gate(user_vecs)
        interest_matrix=refined_interests*(1+gate.unsqueeze(1))

        return interest_matrix

    def label_aware_attention(self,interest_matrix,item_emb):
        if item_emb.dim()==2:
            attn=torch.bmm(interest_matrix,item_emb.unsqueeze(2))/self.dim**0.5
            attn=torch.softmax(attn.squeeze(2),dim=1)
            user_repr=torch.bmm(attn.unsqueeze(1),interest_matrix).squeeze(1)
            score=(user_repr*item_emb).sum(dim=-1)
        else:
            attn=torch.bmm(interest_matrix,item_emb.transpose(1,2))/self.dim**0.5
            attn=torch.softmax(attn,dim=1)
            user_repr=torch.bmm(attn.transpose(1,2),interest_matrix)
            score=(user_repr*item_emb).sum(dim=-1)
        return score

    def diversity_loss(self,interest_matrix):
        norm=interest_matrix/(interest_matrix.norm(dim=-1,keepdim=True)+1e-7)
        sim=torch.bmm(norm,norm.transpose(1,2))
        I=torch.eye(self.n_interests,device=interest_matrix.device).unsqueeze(0).expand(interest_matrix.size(0),-1,-1)
        loss=((sim-I)**2).sum(dim=(1,2)).mean()
        return loss
