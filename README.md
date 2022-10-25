# Teddy study IA prediction

Time series gene expression in TEDDY is collected for a subset of enrolled participants and even those participants have huge amount of data missing. This framework is designed to impute gene expression for all participants, whether they have partially or completely missing gene expression.     

### **Framework**

![Image description](https://github.com/compbiolabucf/Teddy/blob/main/overall_figure.png)

## Required Python packages
- Numpy (>=1.17.2)
- Pandas (>=0.25.1)
- sklearn (>=0.21.3)
- PyTorch (pytorch version >=1.5.0, torchvision version >=0.6.0)

## Imputation

![Image description](https://github.com/compbiolabucf/Teddy/blob/main/Figure_3.png)
**data_processing.py**
Run first to pre-preprocess the gene expression and SNP data for imputation.

*command*: python data_processing.py 

**imputation.py**
Run to perform gene expression imputation on processed data.  

*command*: Python imputation.py --data_directory /home/tanvir/Diabetes/data/processed_data/ --save_directory /home/tanvir/Diabetes/data/imputed_data/ --n_epochs 100 --batch_size 32 --learning_rate 1e-3 --true_enc_hidden_size 100 --true_enc_batch_size 32 --true_enc_num_epochs 100 --true_enc_learning_rate 1e-4 --true_enc_gamma 0.99 --syn_enc_num_epochs 25 --syn_enc_batch_size 512 --syn_enc_learning_rate 1e-5

--data_directory: directory of input data

--save_directory: directory to save output

--n_epochs: number of epoch for C<sub>1</sub>

--batch_size:
--learning_rate:
--true_enc_hidden_size:
--true_enc_batch_size:
--true_enc_num_epochs:
--true_enc_learning_rate:
--true_enc_gamma:
--syn_enc_num_epochs:
--syn_enc_batch_size: 
--syn_enc_learning_rate:

## Prediction




**Prediction.py**
Predicts IA status of participants

## Datasets

Gene expression and SNP can be downloaded from this link:
https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001442.v1.p1
