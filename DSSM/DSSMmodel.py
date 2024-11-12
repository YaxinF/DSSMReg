from __future__ import print_function
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from input import Feat, build_input_features, slice_arrays,combined_dnn_input,Cosine_Similarity
from layers.core import PredictionLayer
from layers.core import DNN
import torch.nn as nn

class BaseTower(nn.Module):
    def __init__(self, TF_dnn_feature_columns, TG_dnn_feature_columns, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=True):
        super(BaseTower, self).__init__()
        torch.manual_seed(seed)

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if self.gpus and str(self.gpus[0]) not in self.device:
            raise ValueError("`gpus[0]` should be the same gpu with `device`")

        self. feature_index = build_input_features(TF_dnn_feature_columns + TG_dnn_feature_columns)


        self.TF_dnn_feature_columns = TF_dnn_feature_columns

        self.TG_dnn_feature_columns = TG_dnn_feature_columns


        self.regularization_weight = []


        self.out = PredictionLayer(task,)
        self.to(device)

        # parameters of callbacks
        self._is_graph_network = True  # used for ModelCheckpoint
        self.stop_training = False # used for EarlyStopping

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.1,
            validation_data=None, shuffle=True, callbacks=None):
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 TGs (x_val, y_val), '
                    'or 3 TGs (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)

            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0 < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))

            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(torch.from_numpy(
            np.concatenate(x, axis=-1)), torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}

            with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                for _, (x_train, y_train) in t:
                    x = x_train.to(self.device).float()
                    y = y_train.to(self.device).float()

                    y_pred = model(x).squeeze()

                    optim.zero_grad()
                    loss = F.binary_cross_entropy(y_pred, y.squeeze(), reduction='sum')
                    reg_loss = self.get_regularization_loss()

                    total_loss = loss + reg_loss + self.aux_loss

                    loss_epoch += loss.item()
                    total_loss_epoch += total_loss.item()
                    total_loss.backward()
                    optim.step()

                    if verbose > 0:
                        for name, metric_fun in self.metrics.items():
                            if name not in train_result:
                                train_result[name] = []
                            train_result[name].append(metric_fun(
                                y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype('float64')
                            ))

            # add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + ": {0: .4f} ".format(epoch_logs[name]) + " - " + \
                                "val_" + name + ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            if self.stop_training:
                break

    def evaluate(self, x, y, batch_size=256):
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256):
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1))
        )
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size
        )

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()

                pred_ans.append(y_pred)
                #print(pred_ans)

        return np.concatenate(pred_ans).astype("float64")

    def input_from_feature_columns(self, X, feature_columns, support_dense=True):
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, Feat), feature_columns)) if len(feature_columns) else []


        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")



        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return dense_value_list



    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer, loss=None, metrics=None):
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1,0))
                self.metrics_names.append(metric)
        return metrics_


class DSSM(BaseTower):
    def __init__(self, TF_dnn_feature_columns, TG_dnn_feature_columns, gamma=1, dnn_use_bn=True,
                 dnn_hidden_units=(300, 300, 128), dnn_activation='leakrelu', l2_reg_dnn=0.001, l2_reg_embedding=1e-6,
                 dnn_dropout=0, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        super(DSSM, self).__init__(TF_dnn_feature_columns, TG_dnn_feature_columns,
                                    l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                    device=device, gpus=gpus)
        #self.output_layer = nn.Linear(dnn_hidden_units[-1], 1)
        #self.sigmoid = nn.Sigmoid()

        if len(TF_dnn_feature_columns) > 0:
            self.TF_dnn = DNN(100, dnn_hidden_units,
                                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.TF_dnn_embedding = None
        if len(TG_dnn_feature_columns) > 0:
            self.TG_dnn = DNN(100, dnn_hidden_units,
                                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.TG_dnn_embedding = None

        self.gamma = gamma
        self.l2_reg_embedding = l2_reg_embedding
        self.seed = seed
        self.task = task
        self.device = device
        self.gpus = gpus

    def forward(self, inputs):
        if len(self.TF_dnn_feature_columns) > 0:
            TF_dense_value_list = \
                self.input_from_feature_columns(inputs, self.TF_dnn_feature_columns)

            TF_dnn_input = combined_dnn_input(TF_dense_value_list)
            self.TF_dnn_embedding = self.TF_dnn(TF_dnn_input)

        if len(self.TG_dnn_feature_columns) > 0:
            TG_dense_value_list = \
                self.input_from_feature_columns(inputs, self.TG_dnn_feature_columns)

            TG_dnn_input = combined_dnn_input(TG_dense_value_list)
            self.TG_dnn_embedding = self.TG_dnn(TG_dnn_input)

        if len(self.TF_dnn_feature_columns) > 0 and len(self.TG_dnn_feature_columns) > 0:
            score = Cosine_Similarity(self.TF_dnn_embedding, self.TG_dnn_embedding, gamma=self.gamma)
            #output = score
            output = self.out(score)
            return output

        elif len(self.TF_dnn_feature_columns) > 0:
            return self.TF_dnn_embedding

        elif len(self.TG_dnn_feature_columns) > 0:
            return self.TG_dnn_embedding

        else:
            raise Exception("input Error! TF and TG feature columns are empty.")


