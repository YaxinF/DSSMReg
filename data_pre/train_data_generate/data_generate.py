import csv
import random
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

# read tf list and tg list
TF_list = []  # 保存列数据的列表
TG_list = []
with open('tf_list.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        TF_list.append(row[0])  # 将第一列数据添加到列表中
with open('tg_list.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        TG_list.append(row[0])  # 将第一列数据添加到列表中

# read chip-seq network
with open('hESC-ChIP-seq-network.csv', 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)  # 将所有行保存到一个列表中

new_rows = []
for row in rows:
    gene1 = row[0]  # 第一列中的基因名
    gene2 = row[1]  # 第二列中的基因名
    if gene1 in TF_list and gene2 in TG_list:
        new_rows.append(row)

# write back
with open('truth.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['TF', 'TG'])  # 写入列名
    writer.writerows(new_rows)

# Generate positive and negative samples
positive_samples = {}  # Dictionary to store positive samples (transcription factor: [target genes])
target_genes = set()  # Set to store all unique target genes

with open('truth.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        tf, target = row
        if tf not in positive_samples:
            positive_samples[tf] = []
        positive_samples[tf].append(target)
        target_genes.add(target)

negative_samples = []

for tf, targets in positive_samples.items():
    other_targets = set(target_genes) - set(targets)
    num_negative_samples = len(targets) * 1

    for _ in range(num_negative_samples):
        if other_targets:
            negative_sample = random.choice(list(other_targets))
            negative_samples.append((tf, negative_sample))
            other_targets.remove(negative_sample)

# Write positive and negative samples alternately to the output CSV file
with open('positive_negative_samples.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['TF', 'Target', 'Label'])  # Header row

    for tf, targets in positive_samples.items():
        for target in targets:
            writer.writerow([tf, target, '1'])  # Write each positive sample with label '1'

    for tf, target in negative_samples:
        writer.writerow([tf, target, '0'])  # Write negative samples with label '0'

# Read gene expression and motif embedding
gene_exp_embedding_path = R"gene_exp.csv"
exp_embedding = open(gene_exp_embedding_path)
exp_embedding_reader = list(csv.reader(exp_embedding))
exp_embedding_record = {}
for exp_single_embedding_reader in exp_embedding_reader[1:]:
    exp_embedding_record[exp_single_embedding_reader[0]] = list(map(float, exp_single_embedding_reader[1:]))

gene_motif_embedding_path = "gene_motif.csv"
motif_embedding = open(gene_motif_embedding_path)
motif_embedding_reader = list(csv.reader(motif_embedding))
motif_embedding_record = {}
for motif_single_embedding_reader in motif_embedding_reader[1:]:
    motif_embedding_record[motif_single_embedding_reader[0]] = list(map(float, motif_single_embedding_reader[1:]))

# Read positive and negative sample data
data_path = "positive_negative_samples.csv"
data = pd.read_csv(data_path)
data = shuffle(data)
split_ratio = 0.8
split_index = int(len(data) * split_ratio)
train = data[:split_index]
test = data[split_index:]

# Get embedding vector
train_TF = list(train['TF'])
train_TG = list(train['Target'])
test_TF = list(test['TF'])
test_TG = list(test['Target'])

exp_embedding_train_TF = [exp_embedding_record[tf] for tf in train_TF]
exp_embedding_train_TG = [exp_embedding_record[tg] for tg in train_TG]
exp_embedding_test_TF = [exp_embedding_record[tf] for tf in test_TF]
exp_embedding_test_TG = [exp_embedding_record[tg] for tg in test_TG]

motif_embedding_train_TF = [motif_embedding_record[tf] for tf in train_TF]
motif_embedding_train_TG = [motif_embedding_record[tg] for tg in train_TG]
motif_embedding_test_TF = [motif_embedding_record[tf] for tf in test_TF]
motif_embedding_test_TG = [motif_embedding_record[tg] for tg in test_TG]

all_embedding_train_TF = [exp + motif for exp, motif in zip(exp_embedding_train_TF, motif_embedding_train_TF)]
all_embedding_train_TG = [exp + motif for exp, motif in zip(exp_embedding_train_TG, motif_embedding_train_TG)]
all_embedding_test_TF = [exp + motif for exp, motif in zip(exp_embedding_test_TF, motif_embedding_test_TF)]
all_embedding_test_TG = [exp + motif for exp, motif in zip(exp_embedding_test_TG, motif_embedding_test_TG)]

# save train and test data
dataframe_train = pd.DataFrame({
    'TF': train_TF,
    'TF_embedding': all_embedding_train_TF,
    'TG': train_TG,
    'TG_embedding': all_embedding_train_TG,
    'score': list(train['Label'])
})
dataframe_test = pd.DataFrame({
    'TF': test_TF,
    'TF_embedding': all_embedding_test_TF,
    'TG': test_TG,
    'TG_embedding': all_embedding_test_TG,
    'score': list(test['Label'])
})

dataframe_train.to_csv(r"train.csv", sep=',', index=False)
dataframe_test.to_csv(r"test.csv", sep=',', index=False)