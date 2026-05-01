import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import DeepFM
import config
from data_loader import DeepFM_Dataset
from metrics import FocalLoss, EarlyStop, ReduceIr
from metrics.evaluate import evaluate_deepfm

device=torch.device(config.device)
torch.manual_seed(config.seed)


def train_deepfm():
    best_auc=0.0
    best_model=None
    early_stop=EarlyStop(p=config.early_stop_p,delta=config.stop_delta)

    history={'loss':[],'auc':[],'accuracy':[],'precision':[],'recall':[],'f1':[]}
    
    with open(config.n_movies_path,'rb') as f:
        n_movies=pickle.load(f)
    with open(config.n_types_path,'rb') as f:
        n_types=pickle.load(f)
    with open(config.user_encoder_path,'rb') as f:
        user_encoder=pickle.load(f)
    n_users=len(user_encoder.classes_)
    
    train_sample=pd.read_csv(config.train_path)
    tag_cols=[col for col in train_sample.columns if col.startswith('tag_')]
    n_tags=len(tag_cols)
    
    model=DeepFM(
        n_users=n_users,
        n_movies=n_movies,
        dim=config.dim,
        n_hour=24,
        n_day=7,
        n_month=12,
        n_types=n_types,
        n_tags=n_tags
    )
    train_dataset=DeepFM_Dataset(data_path=config.train_path)
    val_dataset=DeepFM_Dataset(data_path=config.val_path)

    train_loader=DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    val_loader=DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    model=model.to(device)
    opt=optim.Adam(model.parameters(),lr=config.lr)
    schedule=ReduceIr(
        opti=opt,
        delta=config.stop_delta,
        p=config.early_stop_p,
        reduce_rate=config.reduce_rate
    )
    criterion=FocalLoss(alpha=0.2,gamma=2)
    scaler=torch.amp.GradScaler('cuda')

    for epoch in range(config.epoch):
        model.train()
        total_loss=0.0

        for batch in train_loader:
            user_ids=batch['userid'].to(device).long()
            movie_ids=batch['movieid'].to(device).long()
            labels=batch['label'].to(device).float()
            hour=batch['hour'].to(device).long()
            day=batch['day'].to(device).long()
            month=batch['month'].to(device).long()
            types=batch['types'].to(device).float()
            tags=batch['tags'].to(device).float()

            with torch.amp.autocast('cuda'):
                scores=model(user_ids,movie_ids,hour,day,month,types,tags)
                loss=criterion(scores,labels)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            total_loss+=loss.item()

        average_loss=total_loss/len(train_loader)
        print(f'第{epoch+1}轮的损失为:{average_loss:.5f}')

        metrics=evaluate_deepfm(model,val_loader,device)
        print(f'准确率:{metrics["准确率"]:.4f}')
        print(f'精确率:{metrics["精确率"]:.4f}')
        print(f'召回率:{metrics["召回率"]:.4f}')
        print(f'f1分数:{metrics["f1分数"]:.4f}')
        print(f'auc分数:{metrics["auc分数"]:.4f}')

        history['loss'].append(average_loss)
        history['auc'].append(metrics['auc分数'])
        history['accuracy'].append(metrics['准确率'])
        history['precision'].append(metrics['精确率'])
        history['recall'].append(metrics['召回率'])
        history['f1'].append(metrics['f1分数'])

        schedule.step(metrics['auc分数'])

        if metrics['auc分数']>best_auc:
            best_auc=metrics['auc分数']
            best_model=model.state_dict().copy()
            print(f'最高的auc为:{best_auc:.4f}')

        if early_stop.stop(metrics['auc分数'],model):
            print('早停条件满足，提前结束训练')
            break
    model.load_state_dict(best_model)

    epochs=range(1,len(history['loss'])+1)
    plt.figure(figsize=(12,8))
    plt.rcParams['font.sans-serif']=['SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False

    color_auc='#2ecc71'
    color_acc='#e74c3c'
    color_prec='#3498db'
    color_rec='#f39c12'
    color_f1='#9b59b6'

    ax1=plt.gca()
    line1,=ax1.plot(epochs,history['auc'],color=color_auc,marker='o',linewidth=2,label='AUC')
    ax1.set_xlabel('Epoch',fontsize=12)
    ax1.set_ylabel('AUC',color=color_auc,fontsize=12)
    ax1.tick_params(axis='y',labelcolor=color_auc)
    ax1.set_ylim(0.5,1.0)

    ax2=ax1.twinx()
    line2,=ax2.plot(epochs,history['accuracy'],color=color_acc,marker='s',linewidth=2,label='Accuracy')
    line3,=ax2.plot(epochs,history['precision'],color=color_prec,marker='D',linewidth=2,label='Precision')
    line4,=ax2.plot(epochs,history['recall'],color=color_rec,marker='^',linewidth=2,label='Recall')
    line5,=ax2.plot(epochs,history['f1'],color=color_f1,marker='*',linewidth=2,label='F1')
    ax2.set_ylabel('Accuracy / Precision / Recall / F1',fontsize=12)

    lines=[line1,line2,line3,line4,line5]
    labels=[l.get_label() for l in lines]
    ax1.legend(lines,labels,loc='lower right',fontsize=9)

    plt.title('DeepFM Training Metrics',fontsize=14)
    plt.tight_layout()
    plt.savefig('picture/deepfm_metrics.png',dpi=150)
    plt.close()
    print('训练指标图已保存至 picture/deepfm_metrics.png')

    return model.state_dict(), n_types, n_tags

if __name__=='__main__':
    print('开始训练DeepFM模型')
    trained_model, n_types, n_tags = train_deepfm()
    save_dict={'state_dict': trained_model, 'hyper_params': {
        'dim': config.dim, 'n_types': n_types, 'n_tags': n_tags
    }}
    torch.save(save_dict,config.DeepFM_path)
    print(f'模型已保存至{config.DeepFM_path}')
