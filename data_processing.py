import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from scipy import sparse
from sklearn.decomposition import PCA
import random,os,sys
import argparse
pd.options.mode.chained_assignment = None 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", type=int,help='directory for raw omics data',default='/home/tanvir/Diabetes/data/raw_data/') 
    parser.add_argument("--save_directory", type=int,help='directory to save processed data',default='/home/tanvir/Diabetes/data/processed_data/') 
    args = parser.parse_args()
    print(args)
	
data_directory = args.data_directory
save_directory = args.save_directory
os.makedirs(save_directory, exist_ok=True)


def dbgap_add_subjectage(gene):
	gene_sample_dbgapsub_link = pd.read_csv(data_directory+'phs001562.v1.pht008332.v1.p1.TEDDY_Gene_Expression_Sample.MULTI.txt',delimiter='\t',skiprows=10
              ,usecols=[3,4])
	age_data = pd.read_csv(data_directory+'phs001562.v1.pht008333.v1.p1.c1.TEDDY_Gene_Expression_Sample_Attributes.DS-T1DR-IRB-RD.txt',delimiter='\t',skiprows=10
                ,usecols=[2])
	_,x_ind,y_ind = np.intersect1d(gene.index,gene_sample_dbgapsub_link.iloc[:,1],return_indices=True)
	gene = gene.iloc[x_ind,:]
	gene['subject_id'] = gene_sample_dbgapsub_link.iloc[y_ind,0].values
	gene['age'] = age_data.iloc[y_ind,0].values

	return gene



def dataset(X,maskid):
	#### preparing 3D time series dataset from 2D gene expression 
	month=np.array([3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48])
	data=[]
	for iii,temp in enumerate(maskid):
		temp_data=[]
		temp_X = X[X["subject_id"].isin([temp])]
		for age_ind,temp_age in enumerate(month):
			if temp_age in temp_X.loc[:,'age'].values:
				temp_data.extend(temp_X[temp_X["age"].isin([temp_age])].values)
			else:
				temp_data.extend(np.zeros((1,np.size(temp_X.columns))))

		data.append(temp_data)        
		
	data = np.array(data).astype(np.float32)

	return data


def process_gene(gene):

	#converting illumina probe ID into gene names
	gene_anno = pd.read_csv(data_directory+'gene_anno.csv',skiprows=30,delimiter='\t',usecols=[0,12])
	xy,x_ind,y_ind = np.intersect1d(gene.columns.values,gene_anno.loc[:,'ID'],return_indices=True)
	
	gene = gene.iloc[:,x_ind]
	gene_name = gene_anno.iloc[y_ind,:]
	gene.columns = gene_name.iloc[:,1].values
	gene = gene.groupby(gene.columns, axis=1).sum()

	#only keeping protein coding genes
	genelist = np.unique(pd.read_csv(data_directory+'gencode23.csv',usecols=[10]))
	gene = gene.iloc[:,gene.columns.isin(genelist)]
	gene.dropna(axis='rows',inplace=True)

	#gene expression sample names to their dbgap subject name. 
	#2013 unique gene expsample and 401 unique dbgap subject name
	gene = dbgap_add_subjectage(gene)

	#converting 2D gene exp into 3D time series data
	gene_sample_name = np.unique(gene['subject_id'].values)
	gene = dataset(gene,gene_sample_name)
	gene = gene[:,:,:-2]
	
	niddk_link = pd.read_sas(data_directory+'gene_expression_dbgap_link.sas7bdat')
	xy,x_ind,y_ind = np.intersect1d(gene_sample_name,niddk_link.iloc[:,1],return_indices=True)
	gene = gene[x_ind,:,:]
	gene_sample_name = niddk_link.iloc[y_ind,0].values

	return gene, gene_sample_name


def process_snp(snp):

	snp = snp.toarray()
	pca = PCA(n_components=50)
	snp = pca.fit_transform(snp)

	snp_id = pd.read_csv(data_directory+'phs001037.v2.pht005918.v1.p1.TEDDY_Sample.MULTI.txt',
							skiprows=10,delimiter='\t',usecols=[3])

	link = pd.read_sas(data_directory+'snp_dbgap_link.sas7bdat')
	xy,x_ind,y_ind = np.intersect1d(snp_id,link.iloc[:,1],return_indices=True)
	snp = snp[x_ind,:]
	snp_sample_name = link.iloc[y_ind,0].values

	return snp, snp_sample_name




#preparing gene data
gene = pd.read_csv(data_directory+'teddy_expression_normalized_lumi.txt',delimiter='\t'
			,index_col=0)
gene, gene_sample_name = process_gene(gene)

#preparing snp data
snp = sparse.load_npz(data_directory+'sparse_snp.npz')
snp, snp_sample_name = process_snp(snp)


#some samples are only found in gene expression. 
#separating those samples to be excluded in imputation, included in prediction.

only_ind = ~np.in1d(gene_sample_name,snp_sample_name)
only_in_gene = gene[only_ind,:,:]
only_in_gene_names = gene_sample_name[only_ind]

gene = gene[~only_ind,:,:]
gene_sample_name = gene_sample_name[~only_ind]

zero_matrix = np.zeros((len(snp_sample_name)-len(gene_sample_name),gene.shape[1],gene.shape[2]))
gene = np.concatenate([gene,zero_matrix],0)

missing_names = np.setdiff1d(snp_sample_name,gene_sample_name)
gene_sample_name = np.concatenate([gene_sample_name,missing_names])

sample_name, x_ind, y_ind = np.intersect1d(gene_sample_name, snp_sample_name, return_indices=True)
gene = gene[x_ind,:,:]
snp = snp[y_ind,:]

mask = (gene!=0).astype(int)[:,:,0]
print(np.sum(mask))

np.save(save_directory+'only_in_gene.npy',only_in_gene)
np.save(save_directory+'gene.npy',gene)
np.savetxt(save_directory+'snp.csv',snp,delimiter=',',fmt='%s')
np.savetxt(save_directory+'sample_name.csv',sample_name,delimiter=',',fmt='%s')
np.savetxt(save_directory+'only_in_gene_names.csv',only_in_gene_names,delimiter=',',fmt='%s')
np.savetxt(save_directory+'mask.csv',mask,delimiter=',',fmt='%s')



	
