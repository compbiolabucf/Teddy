# Teddy study IA prediction

Time series gene expression in TEDDY is collected for a subset of enrolled participants and even those participants have huge amount of data missing. This framework is designed to impute gene expression for all participants, whether they have partially or completely missing gene expression.     

### **Framework**

![Image description](https://github.com/compbiolabucf/Teddy/blob/main/Figure_2-1.png)

![Image description](https://github.com/compbiolabucf/Teddy/blob/main/Figure_3.png)

### **User manual**
User manual for this imputation and prediction model is available on github at https://github.com/compbiolabucf/Teddy/blob/main/User%20manual.pdf

### **Example run**
As the TEDDY datasets are protected, we provide dummy datasets to show the workflow of the proposed framework.
Dummy SNP and gene expression datasets can be downloaded from this https://knightsucfedu39751-my.sharepoint.com/:f:/g/personal/t_ahmed_knights_ucf_edu/Eoktd9Y5tUFMs1AWgIgjtLQBSFYbaDcBT0DUDDmLf-2JAg?e=owfxaB


$prediction1.py$ is a simplified version of our prediction algorithm that shows the workflow using only imputed gene expression and SNPs. To go through the actual prediction algorithm please refer to $prediction.py$ which can only be run with TEDDY datasets. The study is designed to solve the limitation of missing values in TEDDY datasets to predict IA. For a more generalized approach involving multi-modal time series and cross-sectional datasets, we have developed another framework downloadable from https://github.com/compbiolabucf/TSEst.


