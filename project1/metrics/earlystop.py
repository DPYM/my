import torch
import numpy as np

class EarlyStop():
    def __init__(self,p=3,delta=0.005):
        self.p=p
        self.best_score=0
        self.count=0
        self.delta=delta
        self.best_model=None

    def stop(self,auc_score,model):
        if self.best_model is None:
            self.best_model={k:v.cpu().clone() for k,v in model.state_dict().items()}
            self.best_score=auc_score
            return False
        
        if auc_score>self.best_score+self.delta:
            self.best_score=auc_score
            self.count=0
            self.best_model={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            self.count+=1
        
        if self.count==self.p:
            print(f'AUC分数连续{self.p}次未上升，触发早停机制')
            model.load_state_dict(self.best_model)
            return True
        else: return False