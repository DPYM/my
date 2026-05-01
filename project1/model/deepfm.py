import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self,n_users,n_movies,dim,hidden_dim=[64,128,64,32],n_hour=24,n_day=7,n_month=12,n_types=20,n_tags=0):
        super().__init__()
        self.user_embedding=nn.Embedding(n_users,dim)
        self.movie_embedding=nn.Embedding(n_movies,dim,padding_idx=0)
        self.hour_embedding=nn.Embedding(n_hour,dim)
        self.day_embedding=nn.Embedding(n_day,dim)
        self.month_embedding=nn.Embedding(n_month,dim)
        self.type_embedding=nn.Linear(n_types,dim,bias=False)
        self.n_tags=n_tags
        if n_tags>0:
            self.tag_embedding=nn.Linear(n_tags,dim,bias=False)

        self.dim=dim
        #FM层
        self.user_linear=nn.Embedding(n_users,1)
        self.movie_linear=nn.Embedding(n_movies,1)
        self.hour_linear=nn.Embedding(n_hour,1)
        self.day_linear=nn.Embedding(n_day,1)
        self.month_linear=nn.Embedding(n_month,1)
        self.type_linear=nn.Linear(n_types,1,bias=False)
        if n_tags>0:
            self.tag_linear=nn.Linear(n_tags,1,bias=False)
        self.bias=nn.Parameter(torch.zeros(1))

        #deep层
        input_dim=dim*(7 if n_tags>0 else 6)
        layer=[]
        pre=input_dim
        for h in hidden_dim:
            layer.append(nn.Linear(pre,h))
            layer.append(nn.ReLU())
            layer.append(nn.Dropout(0.3))
            pre=h
        self.dnn=nn.Sequential(*layer)
        self.dnn_out=nn.Linear(hidden_dim[-1],1)
        self.init_weight()

    def init_weight(self):
        for m in [self.user_embedding,self.movie_embedding,self.user_linear,self.movie_linear,
                  self.hour_embedding,self.day_embedding,self.month_embedding,self.type_embedding,
                  self.hour_linear,self.day_linear,self.month_linear,self.type_linear]:
            nn.init.xavier_uniform_(m.weight)
        if self.n_tags>0:
            nn.init.xavier_uniform_(self.tag_embedding.weight)
            nn.init.xavier_uniform_(self.tag_linear.weight)
        for d in self.dnn:
            if isinstance(d,nn.Linear):
                nn.init.xavier_uniform_(d.weight)
        nn.init.xavier_uniform_(self.dnn_out.weight)
        nn.init.zeros_(self.bias)

    def forward(self,user_ids,movie_ids,hour,day,month,types,tags=None):
        user_vecs=self.user_embedding(user_ids)
        movie_vecs=self.movie_embedding(movie_ids)
        hour_vecs=self.hour_embedding(hour)
        day_vecs=self.day_embedding(day)
        month_vecs=self.month_embedding(month)
        type_vecs=self.type_embedding(types)
        
        #FM一阶项的计算,分别计算用户和电影独立的权重
        fm_first=(self.user_linear(user_ids)+self.movie_linear(movie_ids)+self.hour_linear(hour)
        +self.day_linear(day)+self.month_linear(month)+self.type_linear(types)+self.bias)

        embedding_list=[user_vecs,movie_vecs,hour_vecs,day_vecs,month_vecs,type_vecs]
        if tags is not None and self.n_tags>0:
            tag_vecs=self.tag_embedding(tags)
            embedding_list.append(tag_vecs)
            fm_first=fm_first+self.tag_linear(tags)

        #FM二阶项的计算公式：0.5*(和平方-平方和)
        fm_second_vecs=torch.stack(embedding_list,dim=1)
        sum_square=torch.sum(fm_second_vecs,dim=1)**2
        square_sum=torch.sum(fm_second_vecs**2,dim=1)
        fm_second=0.5*torch.sum(sum_square-square_sum,dim=-1,keepdim=True)

        fm_output=fm_first+fm_second

        #DNN层
        dnn_input=torch.cat(embedding_list,dim=-1)
        dnn_output=self.dnn(dnn_input)
        dnn_output=self.dnn_out(dnn_output).squeeze(-1)

        output=dnn_output+fm_output.squeeze(-1)
        return output