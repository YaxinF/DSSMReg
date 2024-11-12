import numpy as np
import torch
from collections import OrderedDict, namedtuple

DEFAULT_GROUP_NAME = "default_group"

class Feat(namedtuple('Feat', ['name', 'dimension', 'dtype'])):
    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(Feat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()

def build_input_features(feature_columns):
    features = OrderedDict()

    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, Feat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features

def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())

def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)

def combined_dnn_input(dense_value_list):
    if len(dense_value_list) > 0:
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_fun([dense_dnn_input])
    if len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError

def slice_arrays(arrays, start=None, stop=None):
    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None if the value of start '
                         'is a list.')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]
def Cosine_Similarity(query, candidate, gamma=1, dim=-1):
    query_norm = torch.norm(query, dim=dim)
    candidate_norm = torch.norm(candidate, dim=dim)
    cosine_score = torch.sum(torch.multiply(query, candidate), dim=-1)
    cosine_score = torch.div(cosine_score, query_norm*candidate_norm+1e-8)
    cosine_score = torch.clamp(cosine_score, -1, 1.0)*gamma
    return cosine_score

def change_type(X):
    b = np.stack([np.array([float(x) for x in i.replace("[", "").replace("]", "").split(",", 100)]) for i in X], axis=0)
    return b