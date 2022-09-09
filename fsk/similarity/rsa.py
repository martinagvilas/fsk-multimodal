from itertools import product
import pickle

import numpy as np
from scipy.stats import spearmanr

from fsk.config import layers
from fsk.dataprep.utils import get_synsets_ids
from fsk.similarity.sem_distances import get_mcrae_distances
from fsk.similarity.mnet_distances import MultiNetDistance


class RSA():
    def __init__(
        self, project_path, model_1, model_2, feature_type=None
    ):
        self.project_path = project_path
        self.dataset_path = project_path / 'dataset'
        self.results_path = project_path / 'results'
        
        self.models_info = self._get_models_info(model_1, model_2)
        self.synsets_ids, self.concepts = get_synsets_ids(self.dataset_path)
        self.synsets = list(self.synsets_ids.keys())
        self.feature_type = feature_type

    def _get_models_info(self, model_1, model_2):
        models_info = {0: {}, 1:{}}
        for idx, model in enumerate([model_1, model_2]):
            if model == 'sem':
                models_info[idx]['name'] = 'sem_mcrae'
            else:
                models_info[idx]['name'] = model
                model_parts = model.split('_')
                models_info[idx]['dnn'] = model_parts[0]
                models_info[idx]['stream'] = model_parts[1]
                models_info[idx]['layers'] = layers[model]
        return models_info

    def get_distances(self):
        dist = {0: {}, 1: {}}
        labels = {0: {}, 1:{}}
        for idx, model in self.models_info.items():
            if model['name'] == 'sem_mcrae':
                dist[idx], labels[idx] = get_mcrae_distances(
                    self.project_path, self.synsets, self.concepts['mcrae'],
                    self.feature_type
                )
            else:
                dist[idx], labels[idx] = MultiNetDistance(
                    self.project_path, model['dnn'], model['stream'], 
                    model['layers'], self.synsets_ids
                ).get_distances()
        return dist, labels

    def compute(self, save=True):
        # Get distances
        self.dist, self.dist_labels = self.get_distances()
        
        # Filter labels
        self.labels = set(self.dist_labels[0]).intersection(self.dist_labels[1])
        sorted(self.labels)
        for idx, _ in self.models_info.items():
            dist_labels = self.dist_labels[idx]
            label_idxs = [
                i for i, l in enumerate(dist_labels) if l in self.labels
            ]
            for key, val in self.dist[idx].items():
                self.dist[idx][key] = val[label_idxs]
            self.dist_labels[idx] = [
                l for i, l in enumerate(dist_labels) if i in label_idxs
            ]
        assert self.dist_labels[0] == self.dist_labels[1]
        
        # Compute RSA values
        self.rsa_vals = []
        for l1, l2 in product(self.dist[0].keys(), self.dist[1].keys()):
            corr_coef, p_val = spearmanr(
                self.dist[0][l1], self.dist[1][l2], nan_policy='omit'
            )
            self.rsa_vals.append([l1, l2, corr_coef, np.round(p_val, 3)])
        
        # Save and return
        if save == True:
            if self.feature_type == None:
                file = (
                    f"{self.models_info[0]['name']}_"\
                    f"{self.models_info[1]['name']}.pkl"
                )
            else:
                file = (
                    f"{self.models_info[0]['name']}_"\
                    f"{self.models_info[1]['name']}_{self.feature_type}.pkl"
                )
            with open((self.results_path / 'rsa' / file), 'wb') as f:
                pickle.dump(self.rsa_vals, f)
        return self.rsa_vals
