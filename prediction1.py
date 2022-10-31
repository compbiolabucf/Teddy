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
    parser.add_argument("--data_directory", type=str,default='/home/tanvir/Diabetes/') 
    args = parser.parse_args()
    print(args)

data_directory = args.data_directory
device = torch.device(args.device if torch.cuda.is_available() else "cpu")


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


input_data = np.load('gene.npy') #large_cohort
snp_data = pd.read_csv('snp.csv',header=None) # #sparse.load_npz('/data/tahmed/Diabetes/snp_to_gene/gan/snp.npz') #
# generating random labels instead of IA status
y = np.random.choice([0,1],(len(input_data),1),replace=True).astype(np.float32)

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
