import numpy as np
import pandas as pd
from input import Feat,change_type
from DSSMmodel import DSSM
import matplotlib.pyplot as plt
from sklearn import metrics
import os
from sklearn.metrics import *
import argparse

parser = argparse.ArgumentParser(description='Please specify train data, test data,TF vectors data and TG vectors data.')

parser.add_argument('filename1', type=str, help='The name of train data.')
parser.add_argument('filename2', type=str, help='The name of test data.')
parser.add_argument('filename3', type=str, help='The TF vectors data.')
parser.add_argument('filename4', type=str, help='The TG vectors data.')
args = parser.parse_args()

train_file = args.filename1
test_file = args.filename2
TF_file = args.filename3
TG_file = args.filename4

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
TF = pd.read_csv(TF_file, sep=',')
TG = pd.read_csv(TG_file, sep=',')

model_save_dir = os.path.join(os.getcwd(), 'results/')

if __name__ == '__main__':
    features = ['TF_embedding', 'TG_embedding']
    target = "score"

    TF_features = ['TF_embedding']
    TG_features = ['TG_embedding']

    TF_feature_columns = [Feat(feat, 100, ) for feat in TF_features]
    TG_feature_columns = [Feat(feat, 100, ) for feat in TG_features]

    train_model_input = {name: change_type(train[name]) for name in features}
    device = 'cpu'

    model = DSSM(TF_feature_columns, TG_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy", metrics=['auc', 'accuracy'])
    #early_stop = EarlyStopping(monitor='loss', patience=50, mode='min')
    model.fit(train_model_input, train[target].values, batch_size=1024, epochs=100, verbose=2, validation_split=0.2)

    test_model_input = {name: change_type(test[name]) for name in features}

    eval_tr = model.evaluate(train_model_input, train[target].values)
    print(eval_tr)
    pred_ts = model.predict(test_model_input, batch_size=256)
    print(pred_ts)
    print("test LogLoss", round(log_loss(test[target].values, pred_ts), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ts), 4))
    plt.figure(figsize=(10, 6))
    fpr, tpr, thresholds = metrics.roc_curve(test[target].values,pred_ts, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='AUROC = %0.5f)' % auc)
    plt.grid()
    plt.plot([0, 1], [0, 1])
    plt.title('ROC curve')
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.xlim([0, 1])
    plt.legend(loc="lower right")
    print('AUC in figure:', auc)
    plt.savefig("auroc.pdf")
    dict_trained = model.state_dict()
    trained_lst = list(dict_trained.keys())
    # TF tower
    model_TF = DSSM(TF_feature_columns, [],dnn_use_bn=True, task='binary', device=device)
    dict_TF = model_TF.state_dict()
    for key in dict_TF:
        dict_TF[key] = dict_trained[key]
    model_TF.load_state_dict(dict_TF)
    TF_feature_name = TF_features
    TF_model_input = {name: change_type(TF.iloc[:, 1]) for name in TF_feature_name}
    TF_embedding = model_TF.predict(TF_model_input, batch_size=2000)
    print("TF embedding shape: ", TF_embedding[:2])
    df = pd.DataFrame(TF_embedding)
    #Save the TG vector file
    row_names = TF.iloc[:, 0]
    df.index = row_names
    df.to_csv("TF_emb.csv", index=True,header=True)
    # TG tower
    model_TG = DSSM([], TG_feature_columns, dnn_use_bn=True,task='binary', device=device)
    dict_TG = model_TG.state_dict()
    for key in dict_TG:
        dict_TG[key] = dict_trained[key]
    model_TG.load_state_dict(dict_TG)
    TG_feature_name = TG_features
    TG_model_input = {name: change_type(TG.iloc[:, 1]) for name in TG_feature_name}
    TG_embedding = model_TG.predict(TG_model_input, batch_size=2000)
    df2 = pd.DataFrame(TG_embedding)
    row_names2 = TG.iloc[:, 0]
    df2.index = row_names2
    #Save the TG vector file
    df2.to_csv("TG_emb.csv", index=True,header=True)
    print("TG embedding shape: ", TG_embedding[:2])