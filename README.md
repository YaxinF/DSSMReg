# DSSMReg

![image](https://github.com/YaxinF/DSSMReg/blob/main/Fig1.jpg)

DSSMReg builds upon DSSM by utilizing scRNA-seq data, transcription factor motif data, and ChIP-seq data to infer cell-type-specific regulons. It employs AUCell to assess and prioritize the activity of these regulons.  

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
* GSEABase(1.56.0)
# Usage
Here is a step-by-step guide on how to use DSSMReg.

### Expression feature vector generation

Users need to execute the autoencoder.py script to generate gene expression feature vectors. This script inputs gene expression matrix data, creates gene expression feature vectors, and stores them in the gene_exp.csv file. 

`python autoencoder.py <input_data_directory> <output_data_directory>`

The size of the gene expression matrix is C x N, where C represents the number of cells and N represents the number of genes. The format of the expression matrix is as follows. 

            HSPC_025     HSPC_031      HSPC_008    ...    HSPC_026
    CLEC1B  0            0             1.172368    ...    0
    KDM3A   4.891604     6.877725      8.313856    ...    1.105106
    ...     ...          ...           ...         ...    ...
    CORO2B  1.426148     0             1.172368    ...    0

### Motif feature vector generation

Similar to the previous step, Users need to execute the autoencoder.py script to generate motif feature vectors for genes. This script takes a gene motif score matrix file as input, generates motif feature vectors for target genes and stores them in the motif_50.csv file. 

`python autoencoder.py <input_data_directory> <output_data_directory>`

The size of the gene motif score matrix is M x N, where M represents the number of motifs and N represents the number of genes. The format of gene motif score matrix is as follows:

            bergman__Su_H_  cisbp__M00130  c2h2_zfs__M0369  ...    hdpi__ELF2
    CLEC1B  6.44            6.72           12.6             ...    3.43
    KDM3A   4.63            65.1           10.3             ...    5.33
    ...     ...             ...            ...              ...    ...
    CORO2B  6.57            7.55           10.3             ...    3.7

Users need to execute the skip-gram.py to generate motif feature vectors for transcription factors. This script takes a transcription factor motif annotation file as input, generates motif feature vectors for transcription factors, and stores them in the TF_motif_dim_50_iter_9.txt file. 

`python skip-gram.py <input_data_directory> <output_data_directory>`  

The transcription factor motif annotation file contains two columns: the first column is the motif name, and the second column is the transcription factor name. The format is as follows:

     Su_H_     RBPJ     
     M0393     KLF9            
     M00011    ARNt      
     ...       ...           
     MA1483.2  ELF1 

Next, users need to run the gene_motif_generate.py script to generate gene motif feature vectors. The gene_motif_generate.py script combines the previously generated motif feature vectors for target genes (motif_50.csv) and the motif feature vectors of transcription factors (TF_motif_dim_50_iter_9.txt) to generate gene motif feature vectors, which are then stored in the gene_exp.csv file.

 `python gene_motif_generate.py`

### DSSM input and training data generation

Users are required to execute the tf_tg_list_generate.py script to generate the lists of transcription factors and target genes. The tf_tg_list_generate.py script takes the gene list contained in the scRNA-seq data as input and, based on the transcription factor list TF_human.csv, generates separate lists for transcription factors and target genes, saving them in the tf_list.csv and tg_list.csv files, respectively.

`python tf_tg_list_generate.py <gene_list>`

Users are required to execute the Input_generate.py script to generate the input vectors for DSSM. The script combines the expression feature vectors of transcription factors (gene_exp.csv) and motif feature vectors (gene_motif.csv) based on the previously generated transcription factor list (tf_list.csv) and target gene list (tg_list.csv). It generates input vectors for both the transcription factors and target genes, which are then stored in the TF_input.csv and TG_input.csv files, respectively.

`python Input_generate.py`

Users are required to execute the data_generate.py script to generate the training and testing data for DSSM. The Input_generate.py script generates the model's training and testing data based on the ChIP-seq-network file, as well as the expression feature vectors (gene_exp.csv) and motif feature vectors (gene_motif.csv). The generated data are stored in the train.csv and test.csv files.

`python data_generate.py`

The ChIP-seq-network file contains two columns: the first column lists the transcription factor names, and the second column lists the target genes regulated by the transcription factors. The format is as follows:

    Gene1	    Gene2
    AR	    MTRNR2L2
    AR	    MTRNR2L8
    ATF2	    JUN
    ATF2	    CCNI
    ...         ...   
    CREB1	    AKAP1
    CREB1	    GOLGA4

### Embedding generation

Users are required to execute the main.py script to train the DSSM model and generate the embeddings for transcription factors and target genes. The main.py script takes the previously generated input vectors for transcription factors and target genes, as well as the training and testing datasets, as input. It outputs the embeddings for transcription factors and target genes, which are then stored in the TF_emb.csv and TG_emb.csv files, respectively.

`python main.py <train_file><test_file><TF_file><TG_file>`

### Similarity calculation

The Similarity calculation.py script takes the embeddings for transcription factors and target genes(TF_emb.csv and TG_emb.csv), which are output by the DSSM model as input. The Similarity calculation.py script calculates the cosine similarity between each transcription factor and target gene to measure their regulatory relationship. The script then outputs a transcription factor-target gene score matrix stored in the cosine_similarity_matrix.csv file. 

`python Similarity calculation.py`

The transcription factor-target gene score matrix size is T x Z, where T represents the number of target genes, and Z represents the number of transcription factors. The format of the transcription factor-target gene score matrix is as follows: 

    	A1BG                A2M                ...        A4GALT             AAAS
    ABCF2   -0.639341676        -0.85210057        ...        0.745066691        0.778953056
    ACO1	-0.342902508        -0.88357483        ...        0.526061213        0.903887944
    ADARB1	-0.192708076        -0.597843482       ...        0.334189276        0.706811173
    ...     ...                 ...                ...        ...                ...
    AGAP2	-0.982141212        -0.708404815       ...        0.986693573        0.531338671

### Regulon generation

The Regulon generation.py script takes the cosine_similarity_matrix.csv file as input and generates a regulon file in .txt format for each transcription factor. These files are stored in the directory specified by users. Each regulon file is named after the corresponding transcription factor.

`python Regulon generation.py <output_path>`

### AUCell

The AUCell.R script takes the scRNA-seq file and regulon files as input. Users need to specify the path to the scRNA-seq file and the directory where the regulon files are stored in the AUCell.R script. The script calculates the activity of each regulon in each cell. It prioritizes the regulons to specify the path to the scRNA-seq file and the directory where the regulon files are stored in the AUCell.R script. The script calculates the activity of each regulon in each cell, prioritizes the regulons, and stores the results in the aucell.csv file. 

The aucell.csv file stores the activity of regulons, with columns representing cells and values indicating the activity of each regulon in the corresponding cell. The last two columns of the file represent the average activity of each regulon across all cells and the prioritization of regulons, respectively. The format is as follows: 

                     HSC            HSC.1          HSC.2                 HSC.2322       mean           prioritize
    APEX2.txt        0.021637216    0.021673278    0.042481067    ...    0.020122611    0.039571616    ZBED1.txt
    AR.txt	         0.000147771	0.001970282    0              ...    0              0.002872734    GOT1.txt
    ARNT.txt         0.008412072	0.006790788    0.0091164      ...    0.01586732     0.011282365    AR.txt
    ...              ...            ...            ...            ...    ...            ...            ...
    BCL3.txt         0.017045111	0.023451711    0.012356783    ...    0.022535523    0.018944759    ZNF354C.txt





