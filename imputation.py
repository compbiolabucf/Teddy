import numpy as np 
import pandas as pd 
import sys,os,argparse
from models import model


parser = argparse.ArgumentParser()
parser.add_argument("--data_directory", type=str, default='/home/tanvir/Diabetes/data/processed_data/')
parser.add_argument("--save_directory", type=str, default='/home/tanvir/Diabetes/data/imputed_data/')
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=1)
parser.add_argument("--true_enc_hidden_size", type=int, default=100)
parser.add_argument("--true_enc_batch_size", type=int, default=32)
parser.add_argument("--true_enc_num_epochs", type=int, default=100)
parser.add_argument("--true_enc_learning_rate", type=float, default=1e-4)
parser.add_argument("--true_enc_gamma", type=float, default=0.99)
parser.add_argument("--syn_enc_num_epochs", type=int, default=25)
parser.add_argument("--syn_enc_batch_size", type=int, default=512)
parser.add_argument("--syn_enc_learning_rate", type=float, default=1e-5)
args = parser.parse_args()
print(args)

os.makedirs(args.save_directory, exist_ok=True)

gene = np.load(args.data_directory+'gene.npy')
snp = pd.read_csv(args.data_directory+'snp.csv',header=None)

gene = gene[:,:,0:2000]
imputed_gene = np.copy(gene)

new_data = None
for serial in range(gene.shape[1]):
	serial=3
	print('Imputing time point', serial)
	gene_temp = gene[:,serial,:].astype(float)
	for_l2 = gene[:,:serial,:].astype(float)
	generated_samples, test_idx = model(gene_temp,snp.values,args,for_l2)

	for e, i in enumerate(test_idx):
		imputed_gene[i,serial,:] = generated_samples[e,:]

np.save(args.save_directory+'imputed_gene.npy',imputed_gene)	


	
