import torch.nn as nn

class ReduceIr:
    def __init__(self,opti,delta,p,reduce_rate):
        self.opti=opti
        self.delta=delta
        self.p=p
        self.reduce_rate=reduce_rate
        self.best_score=None
        self.round=0
        self.min_lr=1e-5
    
    def step(self,auc):
        if self.best_score==None:
            self.best_score=auc
            return False
        
        if auc<=self.best_score+self.delta:
            self.round+=1
        else:
            self.best_score=auc
            self.round=0
        
        if self.round==self.p:
            self.reduce_lr()
            self.round= 0
        
    def reduce_lr(self):
        for group in self.opti.param_groups:
            old_lr=group['lr']
            new_lr=max(old_lr*self.reduce_rate,self.min_lr)
            group['lr']=new_lr
            print(f'学习率调整,由{old_lr:.5f}下降为{new_lr:.5f}')