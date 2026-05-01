import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
from model import Mind
from data_loader import Mind_Dataset
from metrics import InfoNCELoss, EarlyStop, ReduceIr
from metrics.evaluate import evaluate_mind

device=torch.device(config.device)
torch.manual_seed(config.seed)


def train_mind():
    best_metric=0.0
    best_model=None
    early_stop=EarlyStop(p=config.early_stop_p,delta=config.stop_delta)

    history={'loss':[],'recall':[],'ndcg':[]}

    with open(config.user_history_path,'rb') as f:
        user_history=pickle.load(f)
    with open(config.user_encoder_path,'rb') as f:
        user_encoder=pickle.load(f)
    n_users=len(user_encoder.classes_)
    with open(config.n_movies_path,'rb') as f:
        n_movies=pickle.load(f)

    val_df=pd.read_csv(config.val_path)

    model=Mind(
        n_users,
        n_movies,
        dim=config.dim,
        n_interest=config.n_interest,
        route=config.route,
        dropout=config.mind_dropout
    )

    train_dataset=Mind_Dataset(
        data_path=config.train_path,
        user_history_path=config.user_history_path,
        n_movies=n_movies,
        n_neg=config.n_neg_mind
    )
    train_loader=DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    model=model.to(device)
    opt=optim.Adam(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    schedule=ReduceIr(
        opti=opt,
        delta=config.stop_delta,
        p=config.early_stop_p,
        reduce_rate=config.reduce_rate
    )
    criterion=InfoNCELoss(temperature=0.07)
    scaler=torch.amp.GradScaler('cuda')

    for i in range(config.epoch):
        model.train()
        total_loss=0

        for batch in train_loader:
            user_ids=batch['userid'].to(device).long()
            pos_ids=batch['pos_movie'].to(device).long()
            neg_ids=batch['neg_movie'].to(device).long()
            train_user_history=batch['user_history'].to(device).long()

            with torch.amp.autocast('cuda'):
                interest_matrix=model(user_ids,train_user_history)
                pos_emb=model.movie_embedding(pos_ids)
                neg_emb=model.movie_embedding(neg_ids)

                pos_score=model.label_aware_attention(interest_matrix,pos_emb)
                neg_score=model.label_aware_attention(interest_matrix,neg_emb)

                main_loss=criterion(pos_score,neg_score)
                div_loss=model.diversity_loss(interest_matrix)
                loss=main_loss+0.1*div_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            total_loss+=loss.item()

        average_loss=total_loss/len(train_loader)
        print(f'第{i+1}个epoch的损失为:{average_loss:.5f}')

        metrics=evaluate_mind(model,val_df,user_history,n_movies,device)
        print(f'Recall@50：{metrics["Recall@50"]:.4f}')
        print(f'NDCG@50：{metrics["NDCG@50"]:.4f}')
        print(f'Hit@50：{metrics["Hit@50"]:.4f}')

        history['loss'].append(average_loss)
        history['recall'].append(metrics['Recall@50'])
        history['ndcg'].append(metrics['NDCG@50'])

        monitor_metric=metrics['Recall@50']
        schedule.step(monitor_metric)

        if monitor_metric>best_metric:
            best_metric=monitor_metric
            best_model=model.state_dict().copy()
            print(f'最高Recall@50: {best_metric:.4f}')

        if early_stop.stop(monitor_metric,model):
            print('早停条件满足，提前结束训练')
            break

    model.load_state_dict(best_model)

    epochs=range(1,len(history['loss'])+1)
    plt.figure(figsize=(12,8))
    plt.rcParams['font.sans-serif']=['SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False

    color_recall='#e74c3c'
    color_ndcg='#3498db'

    ax1=plt.gca()
    line1,=ax1.plot(epochs,history['recall'],color=color_recall,marker='s',linewidth=2,label='Recall@50')
    ax1.set_xlabel('Epoch',fontsize=12)
    ax1.set_ylabel('Recall@50',color=color_recall,fontsize=12)
    ax1.tick_params(axis='y',labelcolor=color_recall)

    ax2=ax1.twinx()
    line2,=ax2.plot(epochs,history['ndcg'],color=color_ndcg,marker='^',linewidth=2,label='NDCG@50')
    ax2.set_ylabel('NDCG@50',color=color_ndcg,fontsize=12)
    ax2.tick_params(axis='y',labelcolor=color_ndcg)

    lines=[line1,line2]
    labels=[l.get_label() for l in lines]
    ax1.legend(lines,labels,loc='lower right',fontsize=10)

    plt.title('MIND Training Metrics',fontsize=14)
    plt.tight_layout()
    plt.savefig('picture/mind_metrics.png',dpi=150)
    plt.close()
    print('训练指标图已保存至 picture/mind_metrics.png')

    return model.state_dict()

if __name__=='__main__':
    print('开始训练Mind模型')
    trained_model=train_mind()
    save_dict={'state_dict': trained_model, 'hyper_params': {
        'dim': config.dim, 'n_interest': config.n_interest,
        'route': config.route, 'dropout': config.mind_dropout
    }}
    torch.save(save_dict,config.MIND_path)
    print('训练完成')
