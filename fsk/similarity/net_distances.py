from itertools import combinations
import pickle

import numpy as  np
from scipy.spatial.distance import pdist
import torch

from fsk.dataprep.utils import get_concepts


class NetDistances():
    def __init__(self, project_path, model, stream, layers, hs_type, synset_ids):
        self.dataset_path = project_path / 'dataset'
        self.net_ft_path = project_path / 'results' / model / 'net_ft'
        self.dist_path = project_path / 'results/rsa/distances'
        
        self.model = model
        self.stream = stream
        self.layers = layers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if hs_type != None:
            self.ftype = f'{stream}_{hs_type}'
        else:
            self.ftype = stream
        
        self.labels = list(combinations(list(synset_ids.keys()), 2))
        self.synset_ids = synset_ids
        self.concept_idxs = self.get_concept_idxs()

    def get_concept_idxs(self):
        concepts = get_concepts(self.dataset_path)
        synsets = list(self.synset_ids.keys())
        c_idxs = {}
        found_c = []
        for s, c in zip(synsets, concepts):
            c_count = concepts.count(c)
            if (c_count == 2) & (c in found_c):
                c_idxs[s] = 1
            else:
                c_idxs[s] = 0
            found_c.append(c)
        return c_idxs

    def get_distances(self):
        self.load_distances()
        if len(self.distances) != len(self.layers):
            self.compute_distances()
        return self.distances, self.labels
    
    def load_distances(self):
        self.distances = {}
        labels = {}
        for l in self.layers:
            l_name = f'{self.model}_{self.ftype}-{l}'
            file = self.dist_path / f'{self.model}_{self.ftype}_{l}.pkl'
            if file.is_file():
                with open(file, 'rb') as f:
                    self.distances[l_name], labels[l_name] = pickle.load(f)
        for l, lb in labels.items():
            assert lb == self.labels, "Loaded labels of layer {l} do not match"

    def compute_distances(self):
        if (self.stream == 'img') or (self.stream == 'multi'):
            ft = self.load_img_features()
        elif self.stream == 'txt':
            ft = self.load_txt_features()
        
        for l, l_ft in zip(self.layers, ft):
            l_name = f'{self.model}_{self.stream}-{l}'
            if l_name in self.distances.keys():
                continue
            else:
                l_dist = pdist(l_ft, metric='cosine')
                file = self.dist_path / f'{self.model}_{self.ftype}_{l}.pkl'
                with open(file, 'wb') as f:
                    pickle.dump((l_dist, self.labels), f)
                self.distances[l_name] = l_dist
        assert all([d.shape == (58311,) for d in self.distances.values()])
    
    def load_img_features(self):
        hs_ft = []
        for s, s_imgs in self.synset_ids.items():
            hs_ft.append(self.load_synset_img_features(s, s_imgs, 'hs'))
        hs_ft = torch.stack(hs_ft)
        ft = []
        for _ in range(hs_ft.shape[1]):
            l_ft = np.squeeze(hs_ft[:, 0, :])
            hs_ft = np.delete(hs_ft, 0, axis=1) 
            ft.append(l_ft.detach().numpy())
        del hs_ft
        if 'c-out' in self.layers:
            c_ft = []
            for s_imgs in self.synset_ids.values():
                c_ft.append(self.load_synset_img_features(s, s_imgs, 'c-out'))
            c_ft = torch.stack(c_ft)
            ft.append(c_ft.detach().numpy())
            del c_ft
        assert len(ft) == len(self.layers)
        return ft
    
    def load_synset_img_features(self, synset, s_imgs, layer_type):
        s_ft = []
        for img_id in s_imgs:
            img_ft = torch.load(
                (self.net_ft_path / f'{layer_type}_{self.ftype}_{img_id}.pt'), 
                map_location=torch.device(self.device)
            )
            if self.stream == 'multi':
                img_ft = img_ft[self.concept_idxs[synset], :, :]
            elif (self.stream == 'img') & (layer_type == 'hs'):
                # Only select CLS representation
                img_ft = img_ft[:, 0, :]
            else:
                img_ft = torch.squeeze(img_ft)
            s_ft.append(img_ft)
        s_ft = torch.mean(torch.stack(s_ft), dim=0)
        return s_ft

    def load_txt_features(self):
        hs_file = self.net_ft_path / f'hs_{self.ftype}.pt'
        hs_ft = torch.load(hs_file, map_location=torch.device(self.device))
        ft = []
        for _ in range(hs_ft.shape[1]):
            l_ft = np.squeeze(hs_ft[:, 0, :])
            hs_ft = np.delete(hs_ft, 0, axis=1) 
            ft.append(l_ft.detach().numpy())
        del hs_ft
        if 'c-out' in self.layers:
            out_file = self.net_ft_path / f'c-out_{self.ftype}.pt'
            out_ft = torch.load(out_file, map_location=torch.device(self.device))
            ft.append(out_ft.detach().numpy())
            del out_ft
        assert len(ft) == len(self.layers)
        return ft

