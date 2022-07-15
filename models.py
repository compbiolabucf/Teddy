import numpy as np 
import torch, random, math, copy, sys
from torch import nn, linalg as LA
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch.nn import init
import torch.utils.data
from sklearn.model_selection import train_test_split
from true_encoding import true_encoding
from synthetic_encoding import synthetic_encoding
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self,n_input,hidden_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(), 
            nn.Linear(512, 1024), 
            nn.BatchNorm1d(1024),
           	nn.ReLU(), 
           	nn.Linear(1024, 5096), 
            nn.BatchNorm1d(5096),
            nn.ReLU(), 
            nn.Linear(5096, hidden_size))

    def forward(self, x):
        out = self.model(x)
        return out


def train(data,target,snp,model,optimizer,criterion,args,Z_t,l2,delta):

	model.train()
	
	data = data.to(device)
	target = target.to(device)
	snp = snp.to(device)

	optimizer.zero_grad()
	output = model(data)

	all_output = model(snp).detach()
	Z_hat_t = synthetic_encoding(torch.transpose(all_output,0,1), args)

	l2_loss = torch.sum(torch.exp(-delta)*torch.sum((output-l2)**2,1)) / (np.sum(torch.exp(-delta).cpu().numpy()!=0)*l2.shape[1] + 0.001)
	train_loss = criterion(output,target) + criterion(Z_hat_t,Z_t)+ 0.0001*l2_loss

	train_loss.backward()
	optimizer.step()

	return train_loss


def evaluate(data,target, model, criterion):
    
	model.eval()

	with torch.no_grad():
		
		data = data.to(device)
		target = target.to(device)

		output = model(data)
		val_loss = criterion(output, target)
	
	return val_loss, output

def l2_processing(data):
	delta = np.ones(len(data)) * np.inf
	l2_data = np.zeros((len(data),data.shape[2]))

	if data.size==0:
		return l2_data, delta

	for i in range(len(data)):
		temp_data = data[i,:,:]
		j = data.shape[1]-1
		serial = data.shape[1]
		while j>=0:
			if np.sum(temp_data[j,:])!=0:
				l2_data[i,:] = temp_data[j,:]
				delta[i] = serial-j
				break
			j-=1
	
	return l2_data, delta


def model(gene,snp,args,for_l2):

	GAMMA = args.gamma
	LR = args.learning_rate
	N_EPOCHS = args.n_epochs
	BATCH_SIZE = args.batch_size 
	HIDDEN_SIZE = args.true_enc_hidden_size

	mask = (gene!=0).astype(int)[:,0]
	test_idx = np.where(mask==0)[0]
	train_val_ind = np.setdiff1d(np.arange(len(mask)), test_idx) 
	
	train_idx, val_idx = train_test_split(train_val_ind, train_size=0.8)
	snp = torch.from_numpy(snp).float()
	gene = torch.from_numpy(gene).float()

	X_train = snp[train_idx,:]
	X_val = snp[val_idx,:]
	X_test = snp[test_idx,:]

	y_train = gene[train_idx,:]
	y_val = gene[val_idx,:]
	y_test = gene[test_idx,:]

	for_l2 = for_l2[train_idx,:,:]

	l2, delta = l2_processing(for_l2)
	l2 = torch.from_numpy(l2).float().to(device)
	delta = torch.from_numpy(delta).float().to(device)
	
	#hidden representation of true data
	Z_t = true_encoding(torch.transpose(y_train,0,1), args)

	model = Generator(snp.shape[1],gene.shape[1]).to(device)
	
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))
	
	best_valid_loss = None
	for epoch in range(N_EPOCHS):

		train_loss = train(X_train, y_train, snp, model, optimizer, criterion, args, Z_t, l2, delta)
		valid_loss, _ = evaluate(X_val, y_val, model, criterion)
		
		if best_valid_loss is None or best_valid_loss > valid_loss:
			best_valid_loss = valid_loss
			best_model = copy.deepcopy(model)

		print(f'Epoch: {epoch+1:02}')
		print(f'\tTrain Loss: {train_loss:.3f}')
		print(f'\t Val. Loss: {valid_loss:.3f}')

	_, generated_data = evaluate(X_test, y_test, best_model, criterion)
	
	return generated_data.cpu().numpy(), test_idx


