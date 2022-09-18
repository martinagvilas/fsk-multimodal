from itertools import product
import pickle

import numpy as np
from scipy.stats import spearmanr

from fsk.config import feature_types, models, layers
from fsk.dataprep.utils import get_synsets_ids
from fsk.it_match.load import get_concept_match_distance, get_feature_match_distance
from fsk.similarity.sem_distances import get_mcrae_distances
from fsk.similarity.net_distances import NetDistances


class RSA():
    def __init__(
        self, project_path, model_1, model_2, pred_model=None, 
        pred_type='concepts'
    ):
        self.project_path = project_path
        self.dataset_path = project_path / 'dataset'
        self.results_path = project_path / 'results'
        self.rsa_path = project_path / 'results/rsa/'
        self.rsa_path.mkdir(parents=True, exist_ok=True)
            
        self.pred_model = pred_model
        self.pred_type = pred_type
        self.models_info = self.get_models_info(model_1, model_2)
        
        self.synsets_ids, self.concepts = get_synsets_ids(self.dataset_path)
        self.synsets = list(self.synsets_ids.keys())

    def get_models_info(self, model_1, model_2):
        models_info = {0: {}, 1:{}}
        for idx, model in enumerate([model_1, model_2]):
            if model == 'sem':
                models_info[idx]['name'] = 'sem_mcrae'
            elif model == self.pred_model:
                models_info[idx]['dnn'] = model
                models_info[idx]['name'] = f'{model}_pred'
            else:
                models_info[idx] = models[model]
                models_info[idx]['name'] = model
                models_info[idx]['layers'] = layers[model]
                if models_info[idx]['name'] in (
                    ['clip_txt', 'albef_txt', 'albef_multi', 'vilt_multi']
                ):
                     models_info[idx]['hs_type'] = 'concepts'
                else:
                    models_info[idx]['hs_type'] = None
        return models_info

    def get_distances(self):
        dist = {0: {}, 1: {}}
        labels = {0: {}, 1:{}}
        for idx, model in self.models_info.items():
            if model['name'] == 'sem_mcrae':
                for ft in feature_types:
                    if ft == None:
                        ext = ''
                    else:
                        ext = f'_{ft}'
                    dist[idx][f'sem{ext}'], labels[idx][f'sem{ext}'] = (
                        get_mcrae_distances(
                            self.project_path, self.synsets,
                            self.concepts['mcrae'], ft
                        )
                    )
            elif self.pred_model == model['dnn']:
                m = model['name']
                self.models_info[idx]['name'] = f'{m}_{self.pred_type}'
                if self.pred_type == 'concepts':
                    dist[idx][m], labels[idx][m] = get_concept_match_distance(
                        self.results_path, model['dnn'], self.synsets_ids
                    )
                elif self.pred_type == 'features':
                    dist[idx][m], labels[idx][m] = get_feature_match_distance(
                        self.results_path, model['dnn'], self.synsets_ids
                    )
            else:
                dist[idx], labels[idx] = NetDistances(
                    self.project_path, model['dnn'], model['stream'], 
                    model['layers'], model['hs_type'], self.synsets_ids
                ).get_distances()
        return dist, labels

    def compute(self):
        # Get distances
        self.dist, self.dist_labels = self.get_distances()

        # Compute RSA values
        self.rsa_vals = []
        for l1, l2 in product(self.dist[0].keys(), self.dist[1].keys()):
            # Filter labels if necessary
            m1_labels = self.dist_labels[0][l1]
            m2_labels = self.dist_labels[1][l2]
            if  m1_labels == m2_labels:
                m1_data = self.dist[0][l1]
                m2_data = self.dist[1][l2]
            else:
                rsa_labels = sorted(list(set(m1_labels).intersection(m2_labels)))
                m1_idxs = [m1_labels.index(l) for l in rsa_labels]
                m1_data = self.dist[0][l1][m1_idxs]
                m2_idxs = [m2_labels.index(l) for l in rsa_labels]
                m2_data = self.dist[1][l2][m2_idxs]
            # Compute spearman correlation
            corr_coef, p_val = spearmanr(m1_data, m2_data, nan_policy='omit')
            print(
                f'corr of {l1} and {l2} is {np.round(corr_coef, 4)} '\
                f'and pval of {np.round(p_val, 4)}', flush=True
            )
            self.rsa_vals.append([l1, l2, corr_coef, np.round(p_val, 3)])
        
        # Save and return
        file = (
            f"{self.models_info[0]['name']}_{self.models_info[1]['name']}.pkl"
        )
        with open((self.results_path / 'rsa' / file), 'wb') as f:
            pickle.dump(self.rsa_vals, f)
        return self.rsa_vals
