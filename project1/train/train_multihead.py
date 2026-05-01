import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Multihead_interest
from data_loader import Multihead_Dataset
from metrics import EarlyStop, ReduceIr, InfoNCELoss
from metrics.evaluate import evaluate_multihead
import config

device=torch.device(config.device)
torch.manual_seed(config.seed)


def train_Multihead():
    best_auc=0
    best_model=None
    early_stop=EarlyStop(p=config.early_stop_p,delta=config.stop_delta)

    history={'loss':[],'auc':[],'recall':[],'ndcg':[]}

    with open(config.user_history_path,'rb') as f:
        user_history=pickle.load(f)
    with open(config.user_encoder_path,'rb') as f:
        user_encoder=pickle.load(f)
    n_users=len(user_encoder.classes_)
    with open(config.n_movies_path,'rb') as f:
        n_movies=pickle.load(f)
    
    train_sample=pd.read_csv(config.train_path)
    tag_cols=[col for col in train_sample.columns if col.startswith('tag_')]
    n_tags=len(tag_cols)
    del train_sample

    model=Multihead_interest(
        n_users,
        n_movies,
        dim=config.dim,
        n_interests=config.n_interest,
        n_heads=config.n_heads,
        n_tags=n_tags
    )
    
    train_dataset=Multihead_Dataset(
        data_path=config.train_path,
        user_history_path=config.user_history_path,
    )
    
    train_loader=DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.num_workers,pin_memory=config.pin_memory)
    
    val_df=pd.read_csv(config.val_path)
    
    model=model.to(device)
    opt=optim.Adam(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    schedule=ReduceIr(
        opti=opt,
        delta=config.stop_delta,
        p=config.early_stop_p,
        reduce_rate=config.reduce_rate
    )
    criterion=torch.nn.BCEWithLogitsLoss()
    scaler=torch.amp.GradScaler('cuda')
    
    print(f'开始粗排训练,总共{config.epoch}个epoch')
    for i in range(config.epoch):
        model.train()
        total_loss=0
        
        for batch in train_loader:
            user_ids=batch['userid'].to(device).long()
            movie_ids=batch['movieid'].to(device).long()
            labels=batch['label'].to(device).float()
            batch_history=batch['user_history'].to(device).long()
            batch_history_tags=batch['history_tags'].to(device).float()

            with torch.amp.autocast('cuda'):
                scores=model(user_ids,batch_history,batch_history_tags,movie_ids)
                loss=criterion(scores,labels)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=config.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            
            total_loss+=loss.item()
        
        average_loss=total_loss/len(train_loader)
        print(f'第{i+1}个epoch的损失为:{average_loss:.5f}')

        metrics=evaluate_multihead(model,val_df,user_history,n_movies,n_tags,device)
        print(f'Recall@200：{metrics["Recall@200"]:.4f}')
        print(f'NDCG@200：{metrics["NDCG@200"]:.4f}')
        print(f'Hit@200：{metrics["Hit@200"]:.4f}')
        print(f'auc分数：{metrics["auc分数"]:.4f}')
        print(f'f1分数：{metrics["f1分数"]:.4f}')

        history['loss'].append(average_loss)
        history['auc'].append(metrics['auc分数'])
        history['recall'].append(metrics['Recall@200'])
        history['ndcg'].append(metrics['NDCG@200'])

        schedule.step(metrics['auc分数'])

        if metrics['auc分数']>best_auc:
            best_auc=metrics['auc分数']
            best_model=model.state_dict().copy()
            print(f'最高的auc分数为:{best_auc:.4f}')

        if early_stop.stop(metrics['auc分数'],model):
            print('早停条件满足，提前结束训练')
            break

    model.load_state_dict(best_model)

    epochs=range(1,len(history['loss'])+1)
    plt.figure(figsize=(12,8))
    plt.rcParams['font.sans-serif']=['SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False

    color_auc='#1f77b4'
    color_recall='#ff7f0e'
    color_ndcg='#2ca02c'

    ax1=plt.gca()
    line1,=ax1.plot(epochs,history['auc'],color=color_auc,marker='o',linewidth=2,label='AUC')
    ax1.set_xlabel('Epoch',fontsize=12)
    ax1.set_ylabel('AUC',color=color_auc,fontsize=12)
    ax1.tick_params(axis='y',labelcolor=color_auc)

    ax2=ax1.twinx()
    line2,=ax2.plot(epochs,history['recall'],color=color_recall,marker='s',linewidth=2,label='Recall@200')
    line3,=ax2.plot(epochs,history['ndcg'],color=color_ndcg,marker='^',linewidth=2,label='NDCG@200')
    ax2.set_ylabel('Recall / NDCG',color=color_recall,fontsize=12)
    ax2.tick_params(axis='y',labelcolor=color_recall)

    lines=[line1,line2,line3]
    labels=[l.get_label() for l in lines]
    ax1.legend(lines,labels,loc='lower right',fontsize=10)

    plt.title('Multihead Attention Training Metrics',fontsize=14)
    plt.tight_layout()
    plt.savefig('picture/multihead_metrics.png',dpi=150)
    plt.close()
    print('训练指标图已保存至 picture/multihead_metrics.png')

    return model.state_dict(), n_tags

if __name__=='__main__':
    print('开始训练MultiHeadAttention模型')
    trained_model, n_tags = train_Multihead()
    save_dict={'state_dict': trained_model, 'hyper_params': {
        'dim': config.dim, 'n_interests': config.n_interest,
        'n_heads': config.n_heads, 'n_tags': n_tags
    }}
    torch.save(save_dict,config.Multiheadattention_path)
    print('训练完成!')
