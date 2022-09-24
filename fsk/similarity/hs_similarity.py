import argparse
from pathlib import Path
import pickle

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

from fsk.config import feature_types, layers
from fsk.dataprep.utils import get_synsets_ids
from fsk.it_match.load import get_concept_match_distance
from fsk.similarity.sem_distances import get_mcrae_distances


UNIMODAL_COMPARE = {
    'clip_img': {'img': 'vit_32'}, 
    'clip_txt': {'txt': 'gpt'},
    'albef_img': {'img': 'vit_16'},
    'albef_txt': {'txt': 'bert'},
    'albef_multi': {'img': 'vit_16', 'txt': 'bert'},
    'albef_multi_img': {'img': 'vit_16'},
    'albef_multi_txt': {'txt': 'bert'},
    'vilt_multi': {'img': 'vit_32', 'txt': 'bert'},
    'vilt_multi_img': {'img': 'vit_32'},
    'vilt_multi_txt': {'txt': 'bert'}
}


def compute_hs_similarity(project_path, model, stream):
    synsets_imgs, concepts = get_synsets_ids(project_path / 'dataset')
    hs_sim = {}
    if (stream == 'multi_img') or (stream == 'multi_txt'):
        model_layers = layers[f'{model}_multi']
    else:
        model_layers = layers[f'{model}_{stream}']
    for l in model_layers:
        if stream == 'img':
            file = f'{model}_{stream}_{l}.pkl'
        elif (stream == 'multi_img') or (stream == 'multi_txt'):
            file = f'{model}_multi_concepts_{l}.pkl'
        else:
            file = f'{model}_{stream}_concepts_{l}.pkl'
        with open(project_path / 'results/rsa/distances/' / file, 'rb') as f:
            c_dist, c_labels = pickle.load(f)
        unet_max = {}
        for u_stream, unet in UNIMODAL_COMPARE[f'{model}_{stream}'].items():
            for u_l in layers[unet]:
                uf = project_path / f'results/rsa/distances/{unet}_{u_stream}_{u_l}.pkl'
                with open(uf, 'rb') as f:
                    u_dist, u_labels = pickle.load(f)
                if  c_labels != u_labels:
                    rsa_labels = sorted(list(set(c_labels).intersection(u_labels)))
                    c_idxs = [c_labels.index(u_l) for u_l in rsa_labels]
                    u_idxs = [u_labels.index(u_l) for u_l in rsa_labels]
                    rsa, _ = spearmanr(c_dist[c_idxs], u_dist[u_idxs])
                else:
                    rsa_labels = c_labels
                    rsa, _ = spearmanr(c_dist, u_dist)
                try:
                    if rsa > unet_max[u_stream][1]:
                        unet_max[u_stream] = [u_l, rsa, rsa_labels, u_dist]
                except:
                    unet_max[u_stream] = [u_l, rsa, rsa_labels, u_dist]

        ft_variance = {}
        for ft in feature_types:
            sem_dist, sem_labels = get_mcrae_distances(
                project_path, (synsets_imgs.keys()), concepts['mcrae'], ft
            )
            if  sem_labels != rsa_labels:
                rsa_labels = sorted(list(set(sem_labels).intersection(rsa_labels)))
                sem_idx = [sem_labels.index(l) for l in rsa_labels]
                dist = [sem_dist[sem_idx]]
                for u_stream, _ in UNIMODAL_COMPARE[f'{model}_{stream}'].items():
                    u_idx = [unet_max[u_stream][2].index(l) for l in rsa_labels]
                    dist.append(unet_max[u_stream][-1][u_idx])
                c_idx = [c_labels.index(l) for l in rsa_labels]
                c_d = c_dist[c_idx]
            else:
                dist = [sem_dist]
                for u_stream, _ in UNIMODAL_COMPARE[f'{model}_{stream}'].items(): 
                    dist.append(unet_max[u_stream][-1])
                c_d = c_dist
            
            X = np.column_stack(dist)
            r2 = []
            for idx in [0, 1]:
                r2.append(get_r2(X, c_d, idx))
            ft_variance[ft] = r2[0] - r2[1]
            print(f'{l} {ft}: {ft_variance[ft]}')
        hs_sim[l] = pd.DataFrame.from_dict(ft_variance, orient='index')
    hs_sim = pd.concat(hs_sim)
    hs_sim.to_csv(project_path / 'results/rsa' / f'hs_var_{model}_{stream}.csv')


def get_r2(X, y, idx):
    X = X[:, idx:]
    reg = LinearRegression().fit(X, y)
    r2 = r2_score(y, reg.predict(X))
    return r2


if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    m_help = 'Select which model to run. Can be one of the following options: \
        all, clip, vilt, albef'
    parser.add_argument('-m', action='store', required=True, help=m_help)
    parser.add_argument('-s', action='store', required=True)
    p_help = 'Path to the project folder containing the dataset and source code'
    parser.add_argument('-pp', action='store', required=True, help=p_help)

    model = parser.parse_args().m
    stream =  parser.parse_args().s
    project_path = Path(parser.parse_args().pp)

    compute_hs_similarity(project_path, model, stream)