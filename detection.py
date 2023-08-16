import os
import matplotlib.colors
import torch
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from os.path import dirname
from numpy.linalg import norm, pinv
import torch.backends.cudnn as cudnn
import seaborn as sns
import timm
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.covariance import EmpiricalCovariance
from scipy.special import logsumexp
from sklearn import metrics
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition import FastICA
import torch.utils.model_zoo as model_zoo

os.environ['CUDA_VISIBLE_DEVICES']='0'

__all__ = ['ResNet', 'resnet18', 'resnet50', ]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

device = torch.device('cuda:0')


def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)
    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1
    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)  
    return num_fp, thresh


def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh


def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out


def select_80(data):
    num_rows = data.shape[0]
    num_samples = int(0.8 * num_rows)  

    selected_rows = np.random.choice(num_rows, size=num_samples, replace=False)

    selected_data = data[selected_rows, :]
    return selected_data



def benchmark():
    recall = 0.95
    result = []
    id_data = "imagenet"
    model_arch = 'resnet50'
    feat_path = f"feature/{id_data}/{model_arch}"

    model = timm.create_model("resnet50", pretrained=False)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))


    model.to(device)
    model.eval()

    w = model.fc.weight.cpu().detach().numpy()
    b = model.fc.bias.cpu().detach().numpy()

    feature_id_train = np.load(f"{feat_path}/in_features_train.npy")

    feature_mean = np.mean(feature_id_train, axis=0)

    threshold = np.quantile(feature_id_train, 0.92)
    threshold = 1.0


    # --------------------- Principial component analysis --------------------

    cov = np.cov(feature_id_train.T)
    print('computing the svd results .... ')
    u, s, v = np.linalg.svd(cov)

    k = 256

    M = u[:, :k] @ u[:, :k].T
    dim = M.shape[0]



    feature_id_val = np.load(f"{feat_path}/in_features_test.npy")


    feature_id_clip = np.load(f"{feat_path}/in_features_val.npy").clip(min=None, max=threshold)

    OOD_data_list = ['inat', 'sun50', 'places50', 'dtd']

    feature_oods = {name: np.load(f"{feat_path}/{name}/out_features.npy") for name in OOD_data_list}

    feature_oods_clip = {name: np.load(f"{feat_path}/{name}/out_features.npy").clip(max=threshold) for name in OOD_data_list}


    method = 'Ours'
    logit_id_val = feature_id_val.clip(min=None, max=threshold) @ w.T + b
    rec_norm = np.linalg.norm((feature_id_val - feature_mean) @ (np.identity(dim) - M), axis=-1)
    r = rec_norm / np.linalg.norm(feature_id_clip, axis=-1)

    logit_oods = {name: feat.clip(min=None, max=threshold) @ w.T + b for name, feat in feature_oods.items()}

    score_id = logsumexp(logit_id_val, axis=-1) * (1.0 - r)
    # ------------------------ compute for ood data -------------------------
    for name, logit_ood, feat_ood in zip(OOD_data_list, logit_oods.values(), feature_oods_clip.values()):
        rec_ood = np.linalg.norm((feat_ood - feature_mean) @ (np.identity(dim) - M), axis=-1)
        r_ood = rec_ood / np.linalg.norm(feat_ood, axis=-1)
        score_ood = logsumexp(logit_ood, axis=-1) * (1.0 - r_ood)
        auc_ood, aupr_ood = auc(score_id, score_ood)[0:2]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood, aupr=aupr_ood))
        print(f'{method} - {name} FPR95: {fpr_ood:.2%}, AUROC: {auc_ood:.2%}')

    df = pd.DataFrame(result)
    print(f'{method} - Average FPR95 {df.fpr.mean():.2%}, AUROC {df.auroc.mean():.2%}')
    
if __name__ == '__main__':
    benchmark()
