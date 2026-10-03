import os, json, random, copy
import numpy as np, pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
base='/mnt/data/stage2_work/WILDA_Team_6_Stage_2/Data_Preparation/Training_Testing_Sets'
out='/mnt/data/Stage3/Predictive_Modeling'; os.makedirs(out+'/Results',exist_ok=True); os.makedirs(out+'/Code',exist_ok=True); os.makedirs(out+'/Model',exist_ok=True)
Xtr=pd.read_csv(base+'/X_train.csv'); Xte=pd.read_csv(base+'/X_test.csv'); ytr=pd.read_csv(base+'/y_train.csv')['Churn']; yte=pd.read_csv(base+'/y_test.csv')['Churn']
X_train,X_val,y_train,y_val=train_test_split(Xtr,ytr,test_size=.15,random_state=SEED,stratify=ytr)
X_train=torch.tensor(X_train.values,dtype=torch.float32); y_train=torch.tensor(y_train.values,dtype=torch.float32).view(-1,1)
X_val=torch.tensor(X_val.values,dtype=torch.float32); y_val=torch.tensor(y_val.values,dtype=torch.float32).view(-1,1)
Xt=torch.tensor(Xte.values,dtype=torch.float32)
class ANN(nn.Module):
 def __init__(self,n):
  super().__init__(); self.net=nn.Sequential(nn.Linear(n,32),nn.ReLU(),nn.Dropout(.20),nn.Linear(32,16),nn.ReLU(),nn.Dropout(.20),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,1))
 def forward(self,x): return self.net(x)
model=ANN(Xtr.shape[1]); opt=torch.optim.Adam(model.parameters(),lr=.001); lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]))
best=None; bestvl=1e9; patience=15; wait=0; hist=[]
for epoch in range(1,201):
 model.train(); opt.zero_grad(); logits=model(X_train); loss=lossfn(logits,y_train); loss.backward(); opt.step()
 model.eval()
 with torch.no_grad():
  vl=lossfn(model(X_val),y_val).item()
 hist.append({'epoch':epoch,'train_loss':loss.item(),'val_loss':vl})
 if vl < bestvl-1e-5: bestvl=vl; best=copy.deepcopy(model.state_dict()); bestepoch=epoch; wait=0
 else: wait+=1
 if wait>=patience: break
model.load_state_dict(best); model.eval()
with torch.no_grad(): probs=torch.sigmoid(model(Xt)).numpy().ravel()
pred=(probs>=.5).astype(int); yt=yte.values
metrics={'accuracy':accuracy_score(yt,pred),'precision':precision_score(yt,pred),'recall':recall_score(yt,pred),'f1':f1_score(yt,pred),'roc_auc':roc_auc_score(yt,probs),'best_epoch':bestepoch,'best_validation_loss':bestvl,'test_records':len(yt)}
print(metrics); print(confusion_matrix(yt,pred)); print(classification_report(yt,pred))
# save
ckpt={'model_state_dict':model.state_dict(),'input_features':list(Xtr.columns),'architecture':[16,32,16,8,1],'dropout':0.2,'threshold':0.5,'seed':SEED}
torch.save(ckpt,out+'/Model/trained_ann_model.pt')
pd.DataFrame(hist).to_csv(out+'/Results/training_history.csv',index=False)
pd.DataFrame({'actual_churn':yt,'predicted_probability':probs,'predicted_churn':pred}).to_csv(out+'/Results/churn_predictions.csv',index=False)
with open(out+'/Results/model_metrics.json','w') as f: json.dump(metrics,f,indent=2)
with open(out+'/Results/classification_report.txt','w') as f: f.write(classification_report(yt,pred,digits=4))
pd.DataFrame(confusion_matrix(yt,pred),index=['Actual_No','Actual_Yes'],columns=['Predicted_No','Predicted_Yes']).to_csv(out+'/Results/confusion_matrix.csv')
# permutation importance custom: shuffle each column and measure AUC drop
rng=np.random.default_rng(SEED); baseauc=metrics['roc_auc']; imps=[]
for j,c in enumerate(Xte.columns):
 drops=[]
 for r in range(20):
  arr=Xte.values.copy(); arr[:,j]=rng.permutation(arr[:,j])
  with torch.no_grad(): p=torch.sigmoid(model(torch.tensor(arr,dtype=torch.float32))).numpy().ravel()
  drops.append(baseauc-roc_auc_score(yt,p))
 imps.append((c,float(np.mean(drops)),float(np.std(drops))))
pd.DataFrame(imps,columns=['feature','auc_importance_mean','auc_importance_std']).sort_values('auc_importance_mean',ascending=False).to_csv(out+'/Results/permutation_feature_importance.csv',index=False)
# write runnable script copy
import shutil; shutil.copy('/mnt/data/train_ann_final.py',out+'/Code/train_ann.py')
