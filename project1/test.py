import torch
from torch.utils.data import DataLoader
from model import Mind, DeepFM, Multihead_interest
import pandas as pd
import pickle
import config
from data_loader import DeepFM_Dataset
from metrics.evaluate import evaluate_mind, evaluate_deepfm, evaluate_multihead

device = torch.device(config.device)
torch.manual_seed(config.seed)


def Test(model_name):
    with open(config.user_history_path,'rb') as f:
        user_history=pickle.load(f)
    with open(config.user_encoder_path,'rb') as f:
        user_encoder=pickle.load(f)
    with open(config.n_movies_path,'rb') as f:
        n_movies=pickle.load(f)
    n_users=len(user_encoder.classes_)

    test_df=pd.read_csv(config.test_path)
    tag_cols=[col for col in test_df.columns if col.startswith('tag_')]
    n_tags=len(tag_cols)

    if model_name=='mind':
        ckpt=torch.load(config.MIND_path,map_location=device,weights_only=True)
        hp=ckpt['hyper_params']
        model=Mind(n_users,n_movies,dim=hp['dim'],n_interest=hp['n_interest'],
                   route=hp['route'],dropout=hp['dropout'])
        model.load_state_dict(ckpt['state_dict'])
        model=model.to(device)

        metrics=evaluate_mind(model,test_df,user_history,n_movies,device,K=50)
        print(f'MIND模型测试结果：')
        print(f'Recall@50：{metrics["Recall@50"]:.4f}')
        print(f'NDCG@50：{metrics["NDCG@50"]:.4f}')
        print(f'Hit@50：{metrics["Hit@50"]:.4f}')

    elif model_name=='deepfm':
        with open(config.n_types_path,'rb') as f:
            n_types=pickle.load(f)
        ckpt=torch.load(config.DeepFM_path,map_location=device,weights_only=True)
        hp=ckpt['hyper_params']
        model=DeepFM(n_users,n_movies,dim=hp['dim'],n_types=hp['n_types'],n_tags=hp['n_tags'])
        model.load_state_dict(ckpt['state_dict'])
        model=model.to(device)

        val_dataset=DeepFM_Dataset(data_path=config.test_path)
        val_loader=DataLoader(val_dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.num_workers)

        metrics=evaluate_deepfm(model,val_loader,device)
        print(f'DeepFM模型测试结果：')
        print(f'准确率：{metrics["准确率"]:.4f}')
        print(f'精确率：{metrics["精确率"]:.4f}')
        print(f'召回率：{metrics["召回率"]:.4f}')
        print(f'f1分数：{metrics["f1分数"]:.4f}')
        print(f'auc分数：{metrics["auc分数"]:.4f}')
        print(f'最佳阈值：{metrics["最佳阈值"]:.4f}')

    elif model_name=='multihead':
        ckpt=torch.load(config.Multiheadattention_path,map_location=device,weights_only=True)
        hp=ckpt['hyper_params']
        model=Multihead_interest(n_users,n_movies,dim=hp['dim'],n_interests=hp['n_interests'],
                                 n_heads=hp['n_heads'],n_tags=hp['n_tags'])
        model.load_state_dict(ckpt['state_dict'])
        model=model.to(device)

        metrics=evaluate_multihead(model,test_df,user_history,n_movies,n_tags,device,K=200,n_candidates=1000)
        print(f'Multihead模型测试结果：')
        print(f'Recall@200：{metrics["Recall@200"]:.4f}')
        print(f'NDCG@200：{metrics["NDCG@200"]:.4f}')
        print(f'Hit@200：{metrics["Hit@200"]:.4f}')
        print(f'auc分数：{metrics["auc分数"]:.4f}')
        print(f'f1分数：{metrics["f1分数"]:.4f}')

if __name__=='__main__':
    model_name=input('请输入模型名称：')
    Test(model_name)
