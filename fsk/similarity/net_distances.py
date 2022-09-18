from itertools import combinations
import pickle

from scipy.spatial.distance import pdist
import torch

from fsk.dataprep.utils import get_concepts
from fsk.it_match.load import load_img_net_ft, load_multi_net_ft, load_txt_net_ft


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
        return self.distances, self.dist_labels
    
    def load_distances(self):
        self.distances = {}
        self.dist_labels = {}
        for l in self.layers:
            l_name = f'{self.model}_{self.ftype}-{l}'
            file = self.dist_path / f'{self.model}_{self.ftype}_{l}.pkl'
            if file.is_file():
                with open(file, 'rb') as f:
                    self.distances[l_name], self.dist_labels[l_name] = pickle.load(f)
        # for l, lb in labels.items():
        #     assert lb == self.labels, "Loaded labels of layer {l} do not match"

    def compute_distances(self):
        if self.stream == 'img':
            ft = self.load_img_features()
        elif self.stream == 'txt':
            ft = self.load_txt_features()
        elif self.stream == 'multi':
            ft = self.load_multi_features()
        
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
                self.dist_labels[l_name] = self.labels
        assert all([d.shape == (58311,) for d in self.distances.values()])
    
    def load_img_features(self):
        ft = []
        for l_idx, l in enumerate(self.layers):
            i_ft, _ = load_img_net_ft(
                self.net_ft_path, l, l_idx, self.synset_ids, avg=True
            )
            ft.append(i_ft)
        return ft

    def load_txt_features(self):
        ft = []
        for l_idx, l in enumerate(self.layers):
            ft.append(load_txt_net_ft(
                self.net_ft_path, l, l_idx, hs_type='concept')
            )
        return ft

    def load_multi_features(self):
        ft = []
        for l_idx, l in enumerate(self.layers):
            i_ft, _ = load_multi_net_ft(
                self.net_ft_path, l, l_idx, self.synset_ids, avg=True, 
                hs_type='concept'
            )
            ft.append(i_ft)
        return ft