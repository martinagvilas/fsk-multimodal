from itertools import combinations

import numpy as np
from scipy.spatial.distance import pdist
import torch


def get_match(
    res_path, model, synsets_imgs, center=True, m_type='concept', avg=False, 
    return_np=True
):
    match = []
    labels = []
    for s, s_imgs in synsets_imgs.items():
        if avg == True:
            labels.append(s)
        s_match = []
        for img in s_imgs:
            if avg == False:
                labels.append([s, img])
            img_file = res_path / model / f'{m_type}_match'/ f'{img}.pt'
            i_match = (
                torch.load(img_file, map_location=torch.device('cpu'))
                .to(torch.float64)
            )
            s_match.append(i_match)
        if avg == True:
            s_match = [torch.mean(torch.stack(s_match), dim=0)]
        match = match + s_match
    match = torch.stack(match)
    if center == True:
        match = (match - torch.mean(match)) / torch.std(match)
    if return_np == True:
        match = match.detach().numpy()
    return match, labels
    

def get_concept_match_distance(
    res_path, model, synsets_imgs, center=True, img_to_txt=True, from_preds=True
):
    match, labels = get_match(
        res_path, model, synsets_imgs, center=center, avg=True
    )
    if img_to_txt == False:
        match = match.T
    if from_preds == True:
        dist = match[np.triu_indices_from(match, k=1)]
        dist_labels = list(combinations(labels, 2))
    else:
        dist = pdist(match, 'cosine')
        dist_labels = list(combinations(labels,2))
    return dist, dist_labels


def get_feature_match_distance(res_path, model, synsets_imgs, center=True):
    match, labels = get_match(
        res_path, model, synsets_imgs, center=center, m_type='feature', avg=True
    )
    dist = pdist(match, 'cosine')
    dist_labels = list(combinations(labels,2))
    return dist, dist_labels


def load_img_net_ft(net_ft_path, layer, l_idx, synsets_imgs, avg=False):
    if layer == 'c-out':
        prefix = 'c-out'
    else:
        prefix = 'hs'

    ft = []
    labels = []
    for s, s_imgs in synsets_imgs.items():
        if avg == True:
            labels.append(s)
        f_ft = []
        for img in s_imgs:
            if avg == False:
                labels.append([s, img])
            i_path = net_ft_path / f'{prefix}_img_{img}.pt'
            i_ft = torch.load(i_path, map_location=torch.device('cpu'))
            if layer != 'c-out':
                i_ft = i_ft[l_idx]
            f_ft.append(i_ft)
            del i_ft
        if avg == True:
            f_ft = [torch.mean(torch.stack(f_ft), dim=0)]
        ft = ft + f_ft
    ft = torch.squeeze(torch.stack(ft)).detach().numpy()
    return ft, labels


def load_txt_net_ft(net_ft_path, layer, l_idx, hs_type='concept'):
    if layer == 'c-out':
        prefix = 'c-out'
    else:
        prefix = 'hs'
    
    if hs_type == 'concept':
        file = net_ft_path / f'{prefix}_txt_concepts.pt'
    elif hs_type =='feature':
        file = net_ft_path / f'{prefix}_txt_sem_features.pt'
    ft = torch.load(file, map_location=torch.device('cpu'))
    if layer != 'c-out':
        ft = ft[:, l_idx, :]
    ft = torch.squeeze(ft).detach().numpy()
    return ft


def load_multi_net_ft(
    net_ft_path, layer, l_idx, synsets_imgs, avg=False, hs_type='concept',
    select_idx=True
):
    ft = []
    labels = []
    for s, s_imgs in synsets_imgs.items():
        if avg == True:
            labels.append(s)
        f_ft = []
        for img in s_imgs:
            if avg == False:
                labels.append([s, img])
            if hs_type == 'concept':
                i_path = net_ft_path / f'hs_multi_concepts_{img}.pt'
                sel_idx = list(synsets_imgs.keys()).index(s)
            elif hs_type == 'feature':
                i_path = net_ft_path / f'hs_multi_features_{img}.pt'
            i_ft = torch.load(i_path, map_location=torch.device('cpu'))
            i_ft = i_ft[:, l_idx, :]
            if select_idx == True:
                i_ft = i_ft[sel_idx]
            f_ft.append(i_ft)
            del i_ft
        if avg == True:
            f_ft = [torch.mean(torch.stack(f_ft), dim=0)]
        ft = ft + f_ft
    ft = torch.squeeze(torch.stack(ft)).detach().numpy()
    return ft, labels