from itertools import combinations
import pickle

import numpy as  np
import pandas as pd
from scipy.spatial.distance import pdist
import torch

from fsk.dataprep.mcrae import load_mcrae_info
from fsk.dataprep.utils import get_fsk_features


def get_mcrae_distances(project_path, synsets, concepts, overwrite=False):
    dist_file = project_path / 'results/rsa/distances/mcrae.pkl'
    if (dist_file.is_file()) and (overwrite == False):
        with open(dist_file, 'rb') as f:
            dist, labels = pickle.load(f)
    else:
        sem_matrix = get_mcrae_feature_matrix(project_path, synsets, concepts)
        dist = pdist(sem_matrix.values, metric='cosine')
        labels = list(combinations(sem_matrix.index.tolist(), 2))
        assert len(labels) == len(dist)
        with open(dist_file, 'wb') as f:
            pickle.dump((dist, labels), f)

    return (dist, labels)


def get_mcrae_feature_matrix(project_path, synsets, concepts, overwrite=False):
    feature_matrix_file = project_path/ 'dataset/annotations/mcrae_features.csv'
    if (feature_matrix_file.is_file()) and (overwrite == False):
        f_matrix = pd.read_csv(feature_matrix_file, index_col=0)
    else:
        # Load mcrae feature information
        mcrae_info = load_mcrae_info(project_path/ 'dataset/annotations')
        # Load features of interest
        features = get_fsk_features(project_path / 'dataset')
        # Create feature matrix
        f_matrix =  pd.DataFrame(
            np.zeros((len(synsets), len(features))), 
            index=synsets, columns=features
        )
        for concept, synset in zip(concepts, synsets):
            c_names = concept.split(', ')
            c_info = mcrae_info.loc[mcrae_info['Concept'].isin(c_names)]
            c_info['Feature'] = c_info['Feature'].replace(
                regex=r".+_-_", value=''
            )
            for _, row in c_info.iterrows():
                feature = row['Feature']
                f_matrix.loc[synset, feature] = row['Prod_Freq']
        # Save
        f_matrix.to_csv(feature_matrix_file)
    return f_matrix


def get_visual_net_distances(project_path, model, stream, layers, imgs_ids):

    # Define paths
    net_ft_path = project_path / 'results' / model / 'net_ft'
    dist_path = project_path / 'results/rsa/distances'
    
    # Load hidden states
    hs = [l for l in layers if l.startswith('hs_')]
    dist_files = [dist_path / f'{model}_img_{l}.pkl' for l in hs]
    dist = {}
    for l, file in zip(hs, dist_files):
        if file.is_file():
            with open(file, 'rb') as f:
                dist[l] = pickle.load(f)
    if len(dist) == len(hs):
        hs_compute = False
    else:
        hs_compute = True
    
    # Load contrastive head features
    if 'c_out' in layers:
        c_out_file = dist_path / f'{model}_img_c-out.pkl'
        if c_out_file.is_file():
            with open(c_out_file, 'rb') as f:
                dist['c_out'] = pickle.load(f)
            c_out_compute = False
        else:
            c_out_compute = True
    else:
        c_out_compute = False
      
    # Compute any missing features
    if hs_compute == True:
        file_prefix = net_ft_path / f'hs_{stream}'
        avg_vals = load_net_features(list(imgs_ids.values()), file_prefix)
        dist = []
        for l in hs:
            vals = np.squeeze(avg_vals[:, 0, :])
            avg_vals = np.delete(avg_vals, 0, axis=1)  # delete to free memory
            if l in dist.keys():
                continue
            else: 
                l_dist = pdist(vals, metric='cosine')
                with open(dist_files[l], 'wb') as f:
                    pickle.dump(l_dist, f)
                dist[l] = l_dist
        del avg_vals
        
    # Compute distances of contrastive head
    if c_out_compute  == True:
        file_prefix = net_ft_path / f'c-out_{stream}'
        avg_vals = load_net_features(list(imgs_ids.values()), file_prefix)
        l_dist = pdist(avg_vals, metric='cosine')
        with open(c_out_file, 'wb') as f:
            pickle.dump(l_dist, f)
        dist[l] = l_dist
        del avg_vals

    # Define labels
    labels = list(combinations(list(imgs_ids.keys()), 2))

    return (dist, labels)


def load_net_features(imgs_paths, file_prefix):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avg_vals = []
    for s_imgs in imgs_paths:
        s_vals = []
        for img in s_imgs:
            img_f = f'{file_prefix}_{img}.pt'
            # One file got corrupted during transfer
            try:
                s_vals.append(torch.load(
                    img_f, map_location=torch.device(device)
                ))  
            except:
                continue
        avg_vals.append(torch.mean(torch.stack(s_vals), dim=0))
    avg_vals = torch.stack(avg_vals)
    if len(avg_vals.shape) == 4:
        avg_vals = torch.flatten(avg_vals, start_dim=2, end_dim=3)
    avg_vals = avg_vals.detach().numpy()
    return avg_vals