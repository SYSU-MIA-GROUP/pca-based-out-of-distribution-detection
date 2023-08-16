import timm
import numpy as np
import torch
import os
import argparse
from torch import nn
from tqdm import tqdm
import torch.utils.model_zoo as model_zoo
from dataloader import get_loader_in, get_loader_out

__all__ = ['ResNet', 'resnet18', 'resnet50', ]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

parser = argparse.ArgumentParser(description='Pytorch Feature extract')

parser.add_argument('--dataset', default="imagenet", type=str, help='In-distribution imagenet')
parser.add_argument('--batchsize', default=128, type=int, help='mini-batch size')
parser.add_argument('--model_name', type=str, help='saved model name', default='resnet50')
parser.add_argument('--model_type', default='resnet50', type=str, choices=['resnet18', 'wrn', 'resnet50'])
parser.add_argument('--mode', type=str, help='train or val')
parser.add_argument('--feature_path', type=str, default='feature', help='path for saving the feature matrix')
parser.set_defaults(argument=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = timm.create_model("resnet50", pretrained=False)
model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))


model.to(device)
model.eval()

backbone = nn.Sequential(*list(model.children())[:-1])

OoD_Datas = ['inat', 'sun50', 'places50', 'dtd']


loader_in_dict = get_loader_in(args, split='train')
loaderIn, num_classes = loader_in_dict.train_loader, loader_in_dict.num_classes


def extract(dataname, mode='train', if_ind='True'):
    if if_ind:
        loader_in_dict = get_loader_in(args, split=mode)
        dataloader, num_classes = loader_in_dict.train_loader, loader_in_dict.num_classes
    features_list = []  
    labels_list = [] 
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(dataloader)):
            images = images.to(device)
            labels = labels.item()
            features = backbone(images).cpu().numpy()
            
            features_list.append(features)
            labels_list.append(labels)

        features_matrix = np.concatenate(features_list, axis=0)  
        labels_npy = np.array(labels_list)
        if mode == 'train':
            save_name = 'features_train.npy'
            save_dir = f'{args.feature_path}/{args.dataset}/{args.model_type}'
            label_name = 'labels_train.npy'
            np.save(os.path.join(save_dir, label_name), labels_npy)
        elif mode == 'None':
            save_name = 'out_features.npy'
            save_dir = f'{args.feature_path}/{args.dataset}/{args.model_type}/{dataname}'
        else:
            save_name = 'features_val.npy'
            save_dir = f'{args.feature_path}/{args.dataset}/{args.model_type}'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)  
        np.save(save_path, features_matrix)          

if __name__ == '__main__':
    extract(args.dataset, mode='train')
    extract(args.dataset, mode='val')
    for data in OoD_Datas:
        extract(dataname=data, mode='val', if_ind=False)