import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='Process a input file name.')

parser.add_argument('filename', type=str, help='The name of the all genes in the scRNA-seq data file.')

args = parser.parse_args()


all_gene = args.filename

gene_df = pd.read_csv(all_gene, header=None)

tf_df = pd.read_csv("TF_human.csv", header=None)

# Get a list of all genes and transcription factors
gene_list = gene_df[0].tolist()
TF_ALL_list = tf_df[0].tolist()

# Find the same gene in both lists
tf_list = list(set(gene_list) & set(TF_ALL_list))

# Find target genes that are not on the list of transcription factors
tg_list = list(set(gene_list) - set(tf_list))

pd.DataFrame({'tf': tf_list}).to_csv('tf_list.csv', index=False)

pd.DataFrame({'tg': tg_list}).to_csv('tg_list.csv', index=False)