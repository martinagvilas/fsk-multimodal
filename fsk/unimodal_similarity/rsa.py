from itertools import product
import pickle

import numpy as np
from scipy.stats import spearmanr

from fsk.config import feature_types, layers, uni_models
from fsk.dataprep.utils import get_synsets_ids
from fsk.it_match.load import get_concept_match_average
from fsk.similarity.sem_distances import get_mcrae_distances
from fsk.similarity.net_distances import NetDistances


class RSA():
    def __init__(self, project_path, model):
        self.project_path = project_path
        self.dataset_path = project_path / 'dataset'
        self.net_path = project_path / 'results' / model
        self.rsa_path = project_path / 'results/rsa/concept_pred/'
        
        self.synsets_ids, self.concepts = get_synsets_ids(self.dataset_path)
        self.synsets = list(self.synsets_ids.keys())

        self.model = model

        # Get distances
        self.dist, self.dist_labels = self.get_distances()
        self.pred, self.pred_labels = get_concept_match_average(
            (self.net_path / 'concept_match'), self.synsets_ids
        )

    def get_distances(self):
        dist = {}
        labels = {}
        for ft in feature_types:
            if ft == None:
                ext = ''
            else:
                ext = f'_{ft}'
            dist[f'sem{ext}'], labels[f'sem{ext}'] = get_mcrae_distances(
                self.project_path, self.synsets, self.concepts['mcrae'], ft
            )
        for m in uni_models:
            m_info = uni_models[m]
            dist[m], labels[m] = NetDistances(
                self.project_path, m_info['dnn'], m_info['stream'], 
                layers[m], m_info['hs_type'], self.synsets_ids
            ).get_distances()
        return dist, labels

    def compute(self):
        self.rsa_vals = []
        for um in self.dist.keys():
            # Filter labels if necessary
            if self.dist_labels[um] == self.pred_labels:
                um_data = self.dist[um]
                mm_data = self.pred
            else:
                rsa_labels = list(
                    set(self.dist_labels[um]).intersection(self.pred_labels)
                )
                rsa_labels = sorted(rsa_labels)
                um_idxs = [self.dist_labels[um].index(l) for l in rsa_labels]
                um_data = {}
                for key, val in self.dist[um].items():
                    um_data[key] = val[um_idxs]
                mm_idxs = [self.pred_labels.index(l) for l in rsa_labels]
                mm_data = self.pred[mm_idxs]
            
            # Compute RSA values
            rsa_val = []
            for um_l in um_data.keys():
                corr, p_val = spearmanr(
                    mm_data, um_data[um_l], nan_policy='omit'
                )
                print(f'corr of {self.model} and {um_l} is {corr}')
                rsa_val.append([self.model, um_l, corr, np.round(p_val, 3)])
            self.rsa_vals.append(rsa_val)
            
            # Save and return
            with open((self.rsa_path / f"{self.model}_{um}.pkl"), 'wb') as f:
                pickle.dump(rsa_val, f)
        return self.rsa_vals

    def compute_variance_patitioning(self):
        pass