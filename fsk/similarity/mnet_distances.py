from itertools import combinations
import pickle

import numpy as  np
import pandas as pd
from scipy.spatial.distance import pdist
import torch

from fsk.dataprep.utils import get_concepts


class MultiNetDistance():
    def __init__(self, project_path, model, stream, layers, imgs_ids):
        self.net_ft_path = project_path / 'results' / model / 'net_ft'
        self.dist_path = project_path / 'results/rsa/distances'
        
        self.model = model
        self.layers = layers
        self.stream = stream
        if self.stream == 'multi':
            self.stream_fname = 'multi_concepts'
        else:
            self.stream_fname = self.stream

        self._get_distances()


    def _get_distances(self):
        self._load_distances()
        if len(self.distances_to_compute) > 0:
            self._compute_distances()
    
    
    def _load_distances(self):
        self.distances = {}
        self.distances_to_compute = []
        self.labels = {}
        for l in self.layers:
            file = self.dist_path / f'{self.model}_{self.stream_fname}_{l}.pkl'
            if file.is_file():
                with open(file, 'rb') as f:
                    self.distances[l], self.labels[l] = pickle.load(f)
            else:
                self.distances_to_compute.append(l)

    
    def _compute_distances(self):
        if self.stream == 'img':
            concept_idxs = None
        elif self.stream == 'multi':
            concept_idxs = get_concept_idxs(project_path)
    


def get_img_net_distances(project_path, model, stream, layers, imgs_ids):
    # Define paths
    net_ft_path = project_path / 'results' / model / 'net_ft'
    dist_path = project_path / 'results/rsa/distances'

    # Add concept idxs to stream multimodal and change its name
    if stream == 'img':
        concept_idxs = None
    elif stream == 'multi':
        concept_idxs = get_concept_idxs(project_path)
        stream = 'multi_concepts'
    
    # Compute any missing features
    labels = list(combinations(list(imgs_ids.keys()), 2))
    if hs_compute == True:
        file_prefix = net_ft_path / f'hs_{stream}'
        avg_vals = load_net_features(
            list(imgs_ids.values()), file_prefix, concept_idxs
        )
        dist = _compute_distance(hs, dist, dist_files, labels)
        del avg_vals
        
    # Compute distances of contrastive head
    if c_out_compute == True:
        file_prefix = net_ft_path / f'c-out_{stream}'
        # TODO: get this into right format for distance computation
        avg_vals = np.squeeze(load_net_features(
            list(imgs_ids.values()), file_prefix, concept_idxs
        ))
        dist = _compute_distance(hs, dist, dist_files, labels)
        del avg_vals

    # Assert labels are the same
    assert all(lab == labels for lab in loaded_labels.values())

    return (dist, labels)



def load_net_features(imgs_paths, file_prefix, concept_idxs=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avg_vals = []
    for s_idx, s_imgs in enumerate(imgs_paths):
        s_vals = []
        for img in s_imgs:
            img_f = f'{file_prefix}_{img}.pt'
            # One file got corrupted during transfer
            try:
                img_ft = torch.load(img_f, map_location=torch.device(device))
                if concept_idxs != None:
                    img_ft = img_ft[concept_idxs[s_idx]]
                s_vals.append(img_ft)
            except:
                continue
        avg_vals.append(torch.mean(torch.stack(s_vals), dim=0))
    avg_vals = torch.stack(avg_vals)
    if len(avg_vals.shape) >= 4:
        avg_vals = torch.flatten(avg_vals, start_dim=2, end_dim=-1)
    avg_vals = avg_vals.detach().numpy()
    return avg_vals


def get_concept_idxs(project_path):
    concepts = get_concepts(project_path / 'dataset')
    idxs = []
    same_word = {}
    for c1 in concepts:
        shared_word = []
        for c2 in concepts:
            if c2 in c1:
                shared_word.append(c2)
        idx = [i for i, w in enumerate(shared_word) if w == c1]
        if (len(idx) > 1) & (c1 in same_word.keys()):
            idxs.append(idx[same_word[c1]])
            same_word[c1] += 1
        elif (len(idx) > 1) & (c1 not in same_word.keys()):
            idxs.append(idx[0])
            same_word[c1] = 1
        else:
            idxs.append(idx[0])
    return idxs


def _compute_distance(net_ft, layers, dist_vals, dist_files, labels):
    for l in layers:
        vals = np.squeeze(net_ft[:, 0, :])
        net_ft = np.delete(net_ft, 0, axis=1)  # delete to free memory
        if l in dist_vals.keys():
            continue
        else: 
            l_dist = pdist(vals, metric='cosine')
            l_idx = int(l.split('_')[1])
            with open(dist_files[l_idx], 'wb') as f:
                pickle.dump((l_dist, labels), f)
            dist_vals[l] = l_dist
    return dist_vals


def get_txt_net_distances(project_path, model, layers, synsets):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hs_file = project_path / 'results' / model / f'net_ft/hs_txt_concepts.pt'
    
    vals = torch.load(hs_file, map_location=torch.device(device))
    if model.startswith('clip'):
        vals = torch.flatten(
            torch.permute(vals, (2, 0, 1, 3)), start_dim=2, end_dim=3
        )
    

    # Define labels
    labels = list(combinations(list(synsets, 2)))

    return (dist, labels)