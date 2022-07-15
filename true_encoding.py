import numpy as np 
import torch, random, math, copy, sys
from torch import nn, linalg as LA
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch.nn import init
import torch.utils.data
from sklearn.model_selection import train_test_split
np.random.seed(42)
torch.manual_seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class autoencoder(nn.Module):
    def __init__(self,n_input,hidden_size):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, (n_input//2)*3),
            nn.BatchNorm1d((n_input//2)*3),
            nn.ReLU(),
            nn.Linear((n_input//2)*3, n_input*4),
            nn.BatchNorm1d(n_input*4),
            nn.ReLU(), 
            nn.Linear(n_input*4, (n_input//2)*5), 
            nn.BatchNorm1d((n_input//2)*5),
            nn.ReLU(), 
            nn.Linear((n_input//2)*5, hidden_size))

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, (n_input//2)*5),
            nn.BatchNorm1d((n_input//2)*5),            
            nn.ReLU(),
            nn.Linear((n_input//2)*5, n_input*4),
            nn.BatchNorm1d(n_input*4),            
            nn.ReLU(),
            nn.Linear(n_input*4, (n_input//2)*3),
            nn.BatchNorm1d((n_input//2)*3),            
            nn.ReLU(), 
            nn.Linear((n_input//2)*3, n_input))

    def forward(self, x):
        x_enc = self.encoder(x)
        x = self.decoder(x_enc)
        return x, x_enc


def train(train_loader,model,optimizer,criterion,scheduler):

	model.train()
	loss = 0
	for batch_features in train_loader:
	    
	    batch_features = batch_features.to(device)

	    optimizer.zero_grad()
	    outputs, _ = model(batch_features)
	    train_loss = criterion(outputs, batch_features)

	    train_loss.backward()
	    optimizer.step()

	    loss += train_loss.item()

	scheduler.step()

	return loss/len(train_loader)


def evaluate(iterator, model, criterion):
    
	model.eval()
	loss = 0

	with torch.no_grad():
		for batch_features in iterator:
			
			batch_features = batch_features.to(device)

			outputs, _ = model(batch_features)
			val_loss = criterion(outputs, batch_features)

			loss += val_loss.item()
        
	return loss/len(iterator)


def test(data, model):
    
	model.eval()

	with torch.no_grad():

		data = data.to(device)

		_, Z_enc = model(data)
		
	return Z_enc



def true_encoding(data, args):

	HIDDEN_SIZE = args.true_enc_hidden_size
	BATCH_SIZE = args.true_enc_batch_size
	N_EPOCHS = args.true_enc_num_epochs
	LR = args.true_enc_learning_rate
	GAMMA = args.true_enc_gamma

	all_idx = np.arange(len(data))
	train_idx, val_idx = train_test_split(all_idx, train_size=0.6)

	dataset = [(data[i]) for i in range(len(all_idx))]

	train_sampler = SubsetRandomSampler(train_idx)
	val_sampler = SubsetRandomSampler(val_idx)
	test_sampler = SubsetRandomSampler(all_idx)

	train_loader = torch.utils.data.DataLoader(dataset=dataset,
	                                               batch_size=BATCH_SIZE,
	                                               sampler=train_sampler,
	                                               shuffle=False,
	                                               num_workers=0,
	                                               pin_memory=False)

	val_loader = torch.utils.data.DataLoader(dataset=dataset,
	                                             batch_size=BATCH_SIZE,
	                                             sampler=val_sampler,
	                                             shuffle=False,
	                                             num_workers=0,
	                                             pin_memory=False)

	
	model = autoencoder(data.shape[1],HIDDEN_SIZE).to(device)

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA, last_epoch=-1)
	
	best_valid_loss = None
	for epoch in range(N_EPOCHS):

		train_loss = train(train_loader, model, optimizer, criterion, scheduler)
		valid_loss = evaluate(val_loader, model, criterion)

		if best_valid_loss is None or best_valid_loss > valid_loss:
			best_valid_loss = valid_loss
			best_model = copy.deepcopy(model)

		#print(f'Epoch: {epoch+1:02}')
		#print(f'\tTrain Loss: {train_loss:.3f}')
		#print(f'\t Val. Loss: {valid_loss:.3f}')

	encoding = test(data, best_model)
	
	return encoding