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
Pre-preprocess the gene expression and SNP data for imputation.
**Imputation.py**
Performs gene expression imputation.  

## Prediction




**Prediction.py**
Predicts IA status of participants

## Datasets

Gene expression and SNP can be downloaded from this link:
https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001442.v1.p1
