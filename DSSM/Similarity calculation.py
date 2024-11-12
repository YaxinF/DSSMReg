import pandas as pd
from scipy.spatial import distance
import numpy as np

df1 = pd.read_csv('TF_emb.csv', index_col=0)
df2 = pd.read_csv('TG_emb.csv', index_col=0)

genes1 = df1.index
genes2 = df2.index

# Create a cosine similarity matrix
cosine_similarity_matrix = np.zeros((len(genes1), len(genes2)))

# Calculate the cosine similarity
for i, gene1 in enumerate(genes1):
    for j, gene2 in enumerate(genes2):
        # Calculate the cosine similarity between two gene vectors
        cosine_similarity_matrix[i, j] = 1 - distance.cosine(df1.loc[gene1], df2.loc[gene2])

similarity_df = pd.DataFrame(cosine_similarity_matrix, index=genes1, columns=genes2)

similarity_df.to_csv('cosine_similarity_matrix.csv')