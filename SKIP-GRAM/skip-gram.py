# coding=utf-8
import gensim
import os
import random
import datetime
import generateMatrix as gM
import argparse
parser = argparse.ArgumentParser(description='Process a input file name.')

parser.add_argument('filename', type=str, help='The name of the file to process.')
parser.add_argument('dir1', type=str, help='The source directory path.')
parser.add_argument('dir2', type=str, help='The export  directory path.')
args = parser.parse_args()

fname = args.filename
sourceDir = args.dir1
export_dir = args.dir2

#load data

with open(os.path.join(sourceDir, fname), 'r', encoding='windows-1252') as f:
    gene_pairs = []
    for line in f:
        gene_pair = line.strip().split()
        gene_pairs.append(gene_pair)

current_time = datetime.datetime.now()
print(current_time)
print("shuffle start " + str(len(gene_pairs)))
random.shuffle(gene_pairs)
current_time = datetime.datetime.now()
print(current_time)
print("shuffle done " + str(len(gene_pairs)))

####training parameters########
dimension = 50  # dimension of the embedding
num_workers = 32  # number of worker threads
sg = 1  # sg =1, skip-gram, sg =0, CBOW
max_iter = 10  # number of iterations
window_size = 1  # The maximum distance between the gene and predicted gene within a gene list
txtOutput = True


for current_iter in range(1,max_iter+1):
    if current_iter == 1:
        print("dimension "+ str(dimension) +" iteration "+ str(current_iter)+ " start")
        model = gensim.models.Word2Vec(gene_pairs, size=dimension, window=window_size, min_count=1, workers=num_workers, iter=1, sg=sg)
        model.save(export_dir+"_dim_"+str(dimension)+"_iter_"+str(current_iter))
        if txtOutput:
            gM.outputTxt(export_dir+"_dim_"+str(dimension)+"_iter_"+str(current_iter))
        print("dimension "+ str(dimension) +" iteration "+ str(current_iter)+ " done")
        del model
    else:
        current_time = datetime.datetime.now()
        print(current_time)
        print("shuffle start " + str(len(gene_pairs)))
        random.shuffle(gene_pairs)
        current_time = datetime.datetime.now()
        print(current_time)
        print("shuffle done " + str(len(gene_pairs)))

        print("dimension " + str(dimension) + " iteration " + str(current_iter) + " start")
        model = gensim.models.Word2Vec.load(export_dir+"_dim_"+str(dimension)+"_iter_"+str(current_iter-1))
        model.train(gene_pairs,total_examples=model.corpus_count,epochs=model.iter)
        model.save(export_dir+"_dim_"+str(dimension)+"_iter_"+str(current_iter))
        if txtOutput:
            gM.outputTxt(export_dir+"_dim_"+str(dimension)+"_iter_"+str(current_iter))
        print("dimension " + str(dimension) + " iteration " + str(current_iter) + " done")
        del model