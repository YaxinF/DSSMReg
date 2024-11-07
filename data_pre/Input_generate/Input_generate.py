import csv
import numpy as np

# read tf list and tg list
tf_path = r"tf_list.csv"
with open(tf_path, 'r', newline='') as tf_gene:
    gene_reader = csv.reader(tf_gene)
    tf_list = [single[0] for single in list(gene_reader)[1:]]  # 将reader转换为列表

tg_path = r"tg_list.csv"
with open(tg_path, 'r', newline='') as tg_gene:
    tg_gene_reader = csv.reader(tg_gene)
    tg_list = [single2[0] for single2 in list(tg_gene_reader)[1:]]  # 将reader转换为列表

with open('gene_motif.csv', 'r', newline='') as csvfile, \
        open('gene_exp.csv', 'r') as new_csv_file:
    reader = csv.reader(csvfile)
    new_csv_reader = csv.reader(new_csv_file)
    next(new_csv_reader)  # Skip the header row
    new_csv_embeddings = {row[0]: np.array([float(val) for val in row[1:]]) for row in new_csv_reader}

    tf_output = []
    tg_output = []

    for row in reader:
        gene = row[0]
        if gene in tf_list or gene in tg_list:
            vector = new_csv_embeddings.get(gene)
            if vector is not None:
                appended_vector = np.append(vector, [float(val) for val in row[1:]])
                if gene in tf_list:
                    tf_output.append([gene] + list(appended_vector))
                else:
                    tg_output.append([gene] + list(appended_vector))

# Write TF_OUT_input.csv and TG_OUT_input.csv
with open('TF_input.csv', 'w', newline='') as tf_outfile, \
        open('TG_input.csv', 'w', newline='') as tg_outfile:
    tf_writer = csv.writer(tf_outfile)
    tg_writer = csv.writer(tg_outfile)

    # 写入列名
    tf_writer.writerow(['TF', 'TF_emb'])
    tg_writer.writerow(['TG', 'TF_emb'])

    for output in tf_output:
        merged_row = ','.join(map(str, output[1:]))
        tf_writer.writerow([output[0], merged_row])

    for output in tg_output:
        merged_row = ','.join(map(str, output[1:]))
        tg_writer.writerow([output[0], merged_row])