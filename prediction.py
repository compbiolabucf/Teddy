import pandas as pd
import numpy as np
import sys
from scipy.stats import ttest_ind
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score,classification_report,f1_score
import random
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
import copy
from torch.nn import init
import math
import argparse
from scipy import stats
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

np.set_printoptions(precision=3)
torch.manual_seed(0)
np.random.seed(42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=5)  
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.00001)  
    parser.add_argument("--end", type=int,help='IA cutoff', default=24)  
    parser.add_argument("--serial", type=int,help='gene exp cutoff', default=16) 
    parser.add_argument("--device", type=str,default='cuda') 
    parser.add_argument("--option", type=int,help='0 for combined data, 6 for only gene exp',default=0) 
    parser.add_argument("--demo", type=bool,default=False) 
    parser.add_argument("--HLA", type=bool,default=False) 
    parser.add_argument("--history", type=bool,default=False) 
    parser.add_argument("--imputed_data_dir", type=str,help='save directory for imputation.py',default='/home/tanvir/Diabetes/data/imputed_data/') 
    parser.add_argument("--processed_data_dir", type=str,help='save directory for data_processing.py',default='/home/tanvir/Diabetes/data/processed_data/') 
    parser.add_argument("--raw_data_dir", type=str,help='directory of raw omics data',default='/home/tanvir/Diabetes/data/raw_data/') 
    args = parser.parse_args()
    print(args)

imputed_data_dir = args.imputed_data_dir
processed_data_dir = args.processed_data_dir
raw_data_dir = args.raw_data_dir
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

def persistent_pos(data,mask_id,end):
    new_id = []
    for temp in mask_id:
        y = data[data["MaskID"].isin([temp])]
        y = y.drop_duplicates(subset=['SAMPLE_COLLECTION_VISIT'], keep='first')
        y = y.sort_values(by=['SAMPLE_COLLECTION_VISIT'])
        xy,x_ind,y_ind = np.intersect1d(np.arange(3,end+1,3),y.loc[:,"SAMPLE_COLLECTION_VISIT"].values,return_indices=True)  
        y = y.iloc[y_ind,:]
        #y = np.sort(y[y["OUTCOME"].isin(["b'Pos'"])].loc[:,'SAMPLE_COLLECTION_VISIT'])
        y = y.loc[:,'OUTCOME'].values
        y[y=="b'Neg'"]=0
        y[y=="b'Pos'"]=1

        if np.size(y)>1:
            y = np.convolve(y,np.ones(2,dtype=int),'valid')
            if 2 in y:
                new_id.append(temp)

    return new_id
        

def find_thr(predictions,target):
    best_score=0
    grid = np.arange(0.35,0.95,0.001)
    for thr in grid:
        predictions_b = np.where(predictions>=thr,1,0)
        #score = f1_score(target,predictions_b)
        tn, fp, fn, tp = confusion_matrix(target, predictions_b).ravel()
        score = (tp/(tp+fn))+(tn/(tn+fp))

        if score>best_score:
            best_score = score
            best_thr = thr

    return best_thr


def validation(epoch, loader, model):
    
    # Set model to eval
    model.eval()
    predictions=[]
    target=[]
    with torch.no_grad():
        for x, y in loader:
            
            x = x.to(device)   
            scores = model(x)

            predictions.append(scores.cpu().detach().numpy().ravel().ravel())
            target.append(y.numpy().ravel())

        predictions = np.squeeze(predictions,0)
        target = np.squeeze(target,0)
        threshold = find_thr(predictions,target)
        predictions_b = np.where(predictions>=threshold,1,0)
        tn, fp, fn, tp = confusion_matrix(target, predictions_b).ravel()

    return tp/(tp+fn), threshold

def check_accuracy(epoch, loader, model, threshold):
    
    # Set model to eval
    model.eval()
    predictions=[]
    target=[]
    with torch.no_grad():
        for x, y in loader:
            
            x = x.to(device)   
            scores = model(x)

            predictions.append(scores.cpu().detach().numpy().ravel().ravel())
            target.append(y.numpy().ravel())

        predictions = np.squeeze(predictions,0)
        target = np.squeeze(target,0)
        predictions_b = np.where(predictions>=threshold,1,0)
        tn, fp, fn, tp = confusion_matrix(target, predictions_b).ravel()

    return roc_auc_score(target, predictions),average_precision_score(target, predictions), f1_score(target, predictions_b),\
         tn/(tn+fp), tp/(tp+fn),tp/(tp+fp) , classification_report(target, predictions_b)


def masking(input_data,mask):
    new_data = np.zeros((np.shape(input_data)))

    for row in range(np.size(mask,0)):
        for column in range(np.size(mask,1)):
            if mask[row,column]==0:
                new_data[row,column,:] = input_data[row,column,:]
                input_data[row,column,:]=0

    input_data = new_data.astype(np.float32)
    return input_data


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0)) 

        # Decode the hidden state of the last time step
        out = self.fc(out[:,-1,:])
        out = self.act(out)

        return out



input_data = np.load(imputed_data_dir+'comb_gene_gan.npy') #large_cohort
feature_name = pd.read_csv(processed_data_dir+'feature_name.csv',header=None)
feature_name = feature_name.values
sample_name = pd.read_csv(processed_data_dir+'sample_name.csv',header=None) 
niddk_id = sample_name.values.ravel()

var = np.var(input_data, axis=0)
var = np.mean(var,axis=0)
feat = np.argsort(var)[::-1]

mask = pd.read_csv(processed_data_dir+'mask.csv',header=None,delimiter=',')
input_data = masking(input_data,mask.values)

'''
for i in range(16):
    temp = input_data[:,i,:]
    scaler = QuantileTransformer(n_quantiles=4, random_state=0)#.fit(train.transpose()) #StandardScaler()
    temp = scaler.fit_transform(temp.transpose())
    input_data[:,i,:] = temp.transpose()
'''


clinical_data = pd.read_csv(raw_data_dir+'test_results.csv',index_col=0)
clinical_data = clinical_data[['TEST_NAME','RESULT','OUTCOME','SAMPLE_COLLECTION_VISIT','MaskID']]
clinical_data = clinical_data[clinical_data["OUTCOME"].isin(["b'Neg'","b'Pos'"])] 
clinical_data = clinical_data[clinical_data["MaskID"].isin(niddk_id)] 

total_pp = []
for IA in ["b'GAD'","b'IA2A'","b'MIAA'","b'ZnT8A'"]:
    clinical_data1 = clinical_data[clinical_data["TEST_NAME"].isin([IA])]
    pp = persistent_pos(clinical_data1,niddk_id,args.end)
    total_pp.extend(pp)
    
PP = np.unique(total_pp)

y = np.zeros((np.shape(input_data)[0],1)).astype(np.float32)
for ind, i in enumerate(niddk_id):
    if i in PP:
        y[ind]=1

####

snp = pd.read_csv(raw_data_dir+'small_sparse_matrix.csv',header=None,delimiter=' ') # #sparse.load_npz('/data/tahmed/Diabetes/snp_to_gene/gan/snp.npz') #
snp_data = snp.values.transpose()
snp_sample_name = pd.read_csv(raw_data_dir+'phs001037.v2.pht005918.v1.p1.TEDDY_Sample.MULTI.txt',
            skiprows=10,delimiter='\t',usecols=[3])

link = pd.read_sas(raw_data_dir+'snp_dbgap_link.sas7bdat')
xy,x_ind,y_ind = np.intersect1d(snp_sample_name,link.iloc[:,1],return_indices=True)
snp_sample_name = link.iloc[y_ind,0]

if args.history is True:
    hist = pd.read_sas(raw_data_dir+'family_history.sas7bdat')
    hist1 = hist.filter(regex='^SIBLINGDIABETIC',axis=1)
    hist2 = hist.loc[:,['MOTHERDIABETIC','FATHERDIABETIC','MaskID']]
    hist = pd.concat((hist1,hist2),axis=1)

    snp_sample_name,x_ind,y_ind = np.intersect1d(snp_sample_name,hist.MaskID,return_indices=True)
    snp_data = snp_data[x_ind]
    hist = hist.iloc[y_ind,:]
    ID = []
    for col in hist.columns:
        hist_1 = hist.loc[hist[col].astype(str) == "b'Yes'"]
        ID.extend(hist_1.MaskID)

    fh = np.zeros((np.size(hist.MaskID),1)).astype(np.float32)
    for e,i in enumerate(hist.MaskID):
        if i in ID:
            fh[e]=1

    subset = [2,7,8,9]
    snp_data = snp_data[:,subset]
    fh = np.concatenate((fh,snp_data),axis=1)

else:
    subset = [2,7,8,9]
    fh = snp_data[:,subset]
    

if args.HLA is True:
    HLA = pd.read_csv('/home/tanvir/Diabetes/HLA.csv',header=None)
    snp_sample_name,x_ind,y_ind = np.intersect1d(HLA.iloc[:,0],snp_sample_name,return_indices=True)
    HLA_data = HLA.iloc[x_ind,[1]]
    fh = fh[y_ind,:]
    fh = np.concatenate((fh,HLA_data),axis=1)

niddk_id,x_ind,y_ind = np.intersect1d(niddk_id,snp_sample_name,return_indices=True)
input_data = input_data[x_ind,:,:]
y = y[x_ind] 
fh = fh[y_ind,:]

if args.demo is True:

    ### add demography
    demo = pd.read_csv('/home/tanvir/Diabetes/TEDDY_V20/TEDDY_DATA/screening_form.csv',usecols=[3,11,22,23,24,25,26,27,35])
    demo = demo.set_index(['MaskID'])
    gender = demo.iloc[:,1].astype(str)
    gender[gender=="b'Female'"] = 0
    gender[gender=="b'Male'"] = 1
    #gender[gender=='nan'] = 0

    ethnicity = demo.iloc[:,[0,2,3,4,5,6,7]].astype(str)
    ethnicity[ethnicity== "b'No'"] = 0
    ethnicity[ethnicity== "b'Yes'"] = 1
    ethnicity[ethnicity=='nan'] = 0
    ethnicity[ethnicity=="b'Unknown or not reported'"] = 0
    ethnicity = ethnicity.drop(['RACE_UNKNOWNORNOTREPORTED'],axis=1)

    demography = ethnicity# pd.concat((gender,ethnicity),axis=1,ignore_index=False)
    xy,x_ind,y_ind = np.intersect1d(niddk_id,demography.index,return_indices=True)
    input_data = input_data[x_ind,:]
    fh = fh[x_ind,:]
    y = y[x_ind]
    demography = demography.iloc[y_ind].values.astype(float)

    fh = np.concatenate((fh,demography),axis=1)

fh = np.expand_dims(fh,1)
fh = np.repeat(fh,16,1) 

input_data = input_data[:,:,feat[0:30]]
input_data = np.concatenate([fh,input_data],2).astype(np.float32)
input_data = input_data[:,0:args.serial,:]

print(np.unique(y,return_counts=True))

####
sample_size = np.size(y)
hidden_size = args.hidden_size
num_layers = args.num_layers
num_classes = 1
learning_rate = args.learning_rate
num_epochs = args.num_epochs
batch_size = args.batch_size

for feat in range(15,16):
    AUC=[]
    AUCPR=[]
    F1=[]
    SENSITIVITY=[]
    SPECIFICITY=[]
    PRECISION=[]
    for i in tqdm(range(40,90)):
        ind = np.arange(np.size(y))
        random.seed(i)
        random.shuffle(ind)   
        train_idx = ind[0:math.floor(np.size(y)*.6)]
        val_idx = ind[math.floor(np.size(y)*.6):math.floor(np.size(y)*.8)]
        test_idx = ind[math.floor(np.size(y)*.8):]

        if np.size(np.unique(y[test_idx]))==1:
            continue
        if np.size(np.unique(y[val_idx]))==1:
            continue
        
        input_size = feat - args.option 
        input_data1 = input_data[:,:,args.option:feat]
        input_data1 = torch.tensor(input_data1)
        label=torch.tensor(y)
        
        data_set = [(input_data1[i], label[i]) for i in range(sample_size)]
              
        ind_0=np.where(label[train_idx]==0)[0]
        ind_1=np.where(label[train_idx]==1)[0]  

        ind_0 = np.array(train_idx)[ind_0]
        ind_1 = np.array(train_idx)[ind_1]
        ### oversampling 

        if np.size(ind_0)>np.size(ind_1):
            ind_11=np.random.choice(ind_1,np.size(ind_0))
            new_train_idx = np.concatenate([ind_11,ind_0])
        else:
            ind_00=np.random.choice(ind_0,np.size(ind_1))
            new_train_idx = np.concatenate([ind_00,ind_1])

        #new_train_idx = train_idx
        nearest_multiple = batch_size * math.ceil(len(new_train_idx)/batch_size)
        add = nearest_multiple - len(new_train_idx)
        new_train_idx = np.concatenate((new_train_idx,new_train_idx[0:add]))
        
        train_sampler = SubsetRandomSampler(new_train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                       batch_size=batch_size,
                                                       sampler=train_sampler,
                                                       shuffle=False,
                                                       num_workers=0,
                                                       pin_memory=True)

        val_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                     batch_size=len(val_idx),
                                                     sampler=val_sampler,
                                                     shuffle=False,
                                                     num_workers=0,
                                                     pin_memory=True)

        test_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                      batch_size=len(test_idx),
                                                      sampler=test_sampler,
                                                      shuffle=False,
                                                      num_workers=0,
                                                      pin_memory=True)

    # Initialize network
        model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
        #model = nn.DataParallel(model)
    # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99, last_epoch=-1)

        best_val=None
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(num_epochs):
            model.train()
    
            for batch_idx, (data, targets) in enumerate(train_loader):

                data = data.to(device)
                targets = targets.to(device)
                scores = model(data)

                loss = criterion(scores, targets)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

            val_auc, threshold = validation(epoch, val_loader, model)
            
            if best_val is None:
              is_best = True
              best_val = val_auc
            else:
              is_best = val_auc > best_val
              best_val = max(val_auc, best_val)

            if is_best:  # make a copy of the best model
              model_best = copy.deepcopy(model)
        
            scheduler.step()

        test_auc, test_aucpr, test_f1, test_spec, test_sen, test_pre , cls_rpt= check_accuracy(epoch, test_loader, model_best,threshold)  #
        AUC.append(test_auc)
        AUCPR.append(test_aucpr)
        F1.append(test_f1)
        SPECIFICITY.append(test_spec)
        SENSITIVITY.append(test_sen)
        PRECISION.append(test_pre)
    
     
    print('average AUC:',np.nanmean([AUC]))
    print('average AUCPR:',np.nanmean([AUCPR]))
    print('average F1:',np.nanmean([F1]))
    print('average specificity:',np.nanmean([SPECIFICITY]))
    print('average sensitivity:',np.nanmean([SENSITIVITY]))
    print('average precision:',np.nanmean([PRECISION]))
