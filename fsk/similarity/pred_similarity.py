import argparse
from pathlib import Path
import pickle

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fsk.config import feature_types, layers
from fsk.dataprep.utils import get_synsets_ids
from fsk.it_match.load import get_concept_match_distance
from fsk.similarity.sem_distances import get_mcrae_distances


UNIMODAL_COMPARE = {
    'clip': {'img': 'vit_32', 'txt': 'gpt'}, 
    'albef': {'img': 'vit_16', 'txt': 'bert'}, 
    'vilt': {'img': 'vit_32', 'txt': 'bert'}, 
}
REG_IDXS = {
    'all': [0, 1, 2], 'img_txt': [1, 2]
}

def compute_pred_similarity(project_path, model):
    synsets_imgs, concepts = get_synsets_ids(project_path / 'dataset')

    c_dist, c_labels = get_concept_match_distance(
        (project_path / 'results'), model, synsets_imgs
    )
    ft_variance = {}
    for ft in feature_types:
        sem_dist, sem_labels = get_mcrae_distances(
            project_path, (synsets_imgs.keys()), concepts['mcrae'], ft
        )

        u_max = {}
        for stream, unet in UNIMODAL_COMPARE[model].items():
            sem_idxs = []
            u_idxs = []
            for l in layers[unet]:
                uf = project_path / f'results/rsa/distances/{unet}_{stream}_{l}.pkl'
                with open(uf, 'rb') as f:
                    u_dist, u_labels = pickle.load(f)
                if  (sem_labels != u_labels) & (sem_idxs == []) & (u_idxs == []):
                    rsa_labels = sorted(list(set(sem_labels).intersection(u_labels)))
                    sem_idxs = [sem_labels.index(l) for l in rsa_labels]
                    u_idxs = [u_labels.index(l) for l in rsa_labels]
                    rsa, _ = spearmanr(sem_dist[sem_idxs], u_dist[u_idxs])
                elif sem_labels == u_labels:
                    rsa_labels = sem_labels
                    rsa, _ = spearmanr(sem_dist, u_dist)
                else:
                    rsa, _ = spearmanr(sem_dist[sem_idxs], u_dist[u_idxs])
                try:
                    if rsa > u_max[stream][1]:
                        u_max[stream] = [l, rsa, rsa_labels, u_dist]
                except:
                    u_max[stream] = [l, rsa, rsa_labels, u_dist]

        if  c_labels != rsa_labels:
            rsa_labels = sorted(list(set(c_labels).intersection(rsa_labels)))
            sem_idx = [sem_labels.index(l) for l in rsa_labels]
            sem_d = sem_dist[sem_idx]
            txt_idx = [u_max['txt'][2].index(l) for l in rsa_labels]
            txt_d = u_max['txt'][-1][txt_idx]
            img_idx = [u_max['img'][2].index(l) for l in rsa_labels]
            img_d = u_max['img'][-1][img_idx]
            c_idx = [c_labels.index(l) for l in rsa_labels]
            c_d = c_dist[c_idx]
        else:
            sem_d = sem_dist
            img_d = u_max['img'][-1]
            txt_d = u_max['txt'][-1]
            c_d = c_dist
        
        X = np.column_stack((c_d, img_d, txt_d))
        r2 = {}
        for reg_type, idx in REG_IDXS.items():
            r2[reg_type] = get_r2(X, sem_d, idx)
        ft_variance[ft] = r2['all'] - r2['img_txt']
        print(f'{ft}: {ft_variance[ft]}')
    ft_variance = pd.DataFrame.from_dict(ft_variance, orient='index')
    ft_variance.to_csv(project_path / 'results/rsa' / f'pred_var_{model}.csv')


def get_r2(X, y, idx):
    X = X[:, idx]
    X = StandardScaler().fit_transform(X)
    reg = LinearRegression().fit(X, y)
    r2 = r2_score(y, reg.predict(X))
    return r2


if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    m_help = 'Select which model to run. Can be one of the following options: \
        all, clip, vilt, albef'
    parser.add_argument('-m', action='store', required=True, help=m_help)
    p_help = 'Path to the project folder containing the dataset and source code'
    parser.add_argument('-pp', action='store', required=True, help=p_help)

    model = parser.parse_args().m
    project_path = Path(parser.parse_args().pp)

    compute_pred_similarity(project_path, model)