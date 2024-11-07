# DSSMReg

![image](https://github.com/YaxinF/DSSMReg/blob/main/Fig1.jpg)

DSSMReg is based on DSSM, using scRNA-seq data, transcription factor motif data and ChIP-seq data to infer cell type-specific regulators, and using AUCell to rank the importance of the regulators. 

# Requirements
## Python
* pandas(1.4.3)
* numpy(1.23.0)
* torch(1.12.1)
* tensorflow(2.11.0)
* gensim(3.8.0)
## R
* Seurat(4.3.0)
* dplyr(1.1.0)
* AUCell(1.16.0)
# Usage
Here is a step-by-step guide on how to use DSSMReg:

### Expression feature vector generation

In the Autoencoder.py file, set the input scRNA-seq file (ExpressionData.csv). `python autoencoder.py`

### Motif feature vector generation

In the Autoencoder.py file, set the input gene_motif_score_matrix file to obtain the motif feature vectors for all genes. `python autoencoder.py`

Set the input file (motif2tf_human.txt) in skip-gram.py to get the motif feature vector of the transcription factor. `python skip-gram.py`

gene_motif_generate.py combines the motif feature vectors of the previously generated transcription factors and the target gene feature vectors in the gene_motif.csv file. `python gene_motif_generate.py`

### DSSM input and training data generation

Generate a list of transcription factors and target genes contained in the expression data. `python tf_tg_list_generate.py`

Combine the expression feature vectors of transcription factors and target genes with motif feature vectors to generate the input for DSSM. `python Input_generate.py`

Generate the training and testing datasets. `python data_generate.py`

### Embedding generation

Run the DSSM model to generate the insertion of target genes for transcription factors. `python main.py`

### Similarity calculation

Calculate the cosine similarity between each transcription factor and target gene embeddings, which represents the strength of the regulatory relationship score.`python Similarity calculation.py`

### Regulon generation

Generate regulons based on regulatory score. `python Regulon generation.py`

### AUCell

Use AUCell to infer the activity of regulatory modules and rank them; the code can be found in AUCell.R.
