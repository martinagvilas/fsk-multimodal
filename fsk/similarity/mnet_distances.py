from itertools import combinations
import pickle

import numpy as  np
import pandas as pd
from scipy.spatial.distance import pdist
import torch

from fsk.dataprep.utils import get_concepts


class MultiNetDistance():
    def __init__(self, project_path, model, stream, layers, synset_ids):
        self.net_ft_path = project_path / 'results' / model / 'net_ft'
        self.dist_path = project_path / 'results/rsa/distances'
        
        self.model = model
        self.layers = layers
        self.stream = stream
        if self.stream == 'multi':
            self.stream_fname = 'multi_concepts'
        else:
            self.stream_fname = self.stream
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.labels = list(combinations(list(synset_ids.keys()), 2))
        self.synset_ids = synset_ids

    def get_distances(self):
        self._load_distances()
        if len(self.distances) != len(self.layers):
            self._compute_distances()
        return self.distances, self.labels
    
    def _load_distances(self):
        self.distances = {}
        labels = {}
        for l in self.layers:
            l_name = f'{self.model}_{self.stream}-{l}'
            file = self.dist_path / f'{self.model}_{self.stream_fname}_{l}.pkl'
            if file.is_file():
                with open(file, 'rb') as f:
                    self.distances[l_name], labels[l_name] = pickle.load(f)
        for l, lb in labels.items():
            assert lb == self.labels, "Loaded labels of layer {l} do not match"

    def _compute_distances(self):
        if self.stream == 'img':
            concept_idxs = None
            ft = self._load_img_features(concept_idxs)
        elif self.stream == 'multi':
            concept_idxs = _get_concept_idxs(self.project_path)
            ft = self._load_img_features(concept_idxs)
        elif self.stream == 'txt':
            ft = self._load_txt_features()
        
        for l, l_ft in zip(self.layers, ft):
            l_name = f'{self.model}_{self.stream}-{l}'
            if l_name in self.distances.keys():
                continue
            else:
                l_dist = pdist(l_ft, metric='cosine')
                file = self.dist_path / f'{self.model}_{self.stream_fname}_{l}.pkl'
                with open(file, 'wb') as f:
                    pickle.dump((l_dist, self.labels), f)
                self.distances[l_name] = l_dist
        assert all([d.shape == (58311,) for d in self.distances.values()])
    
    def _load_txt_features(self):
        hs_file = self.net_ft_path / 'hs_txt_concepts.pt'
        hs_ft = torch.load(hs_file, map_location=torch.device(self.device))
        if self.model.startswith('clip'):
            hs_ft = torch.flatten(
                torch.permute(hs_ft, (2, 0, 1, 3)), start_dim=2, end_dim=3
            )
        ft = []
        for _ in range(hs_ft.shape[1]):
            l_ft = np.squeeze(hs_ft[:, 0, :])
            hs_ft = np.delete(hs_ft, 0, axis=1) 
            ft.append(l_ft)
        del hs_ft
        if 'c-out' in self.layers:
            out_file = self.net_ft_path / 'c-out_txt_concepts.pt'
            out_ft = torch.load(out_file, map_location=torch.device(self.device))
            ft.append(out_ft)
        del out_ft
        assert len(ft) == len(self.layers)
        return ft
    


def get_img_net_distances(project_path, model, stream, layers, imgs_ids):

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


def _get_concept_idxs(project_path):
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

