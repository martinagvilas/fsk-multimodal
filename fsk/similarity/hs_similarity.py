import argparse
from pathlib import Path
import pickle

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

from fsk.config import feature_types, layers
from fsk.dataprep.utils import get_synsets_ids
from fsk.similarity.sem_distances import get_mcrae_distances


UNIMODAL_COMPARE = {
    'clip': {'img': 'vit_32', 'txt': 'gpt'}, 
    'albef': {'img': 'vit_16', 'txt': 'bert'}, 
    'vilt': {'img': 'vit_32', 'txt': 'bert'}, 
}

def compute_hs_similarity(project_path, model, stream):
    synsets_imgs, concepts = get_synsets_ids(project_path / 'dataset')
    ft_variance = {}
    for ft in feature_types:
        sem_dist, sem_labels = get_mcrae_distances(
            project_path, (synsets_imgs.keys()), concepts['mcrae'], ft
        )

        u_max = {}
        for ustream, unet in UNIMODAL_COMPARE[model].items():
            max_corr_file =  project_path / 'results/rsa' / f'max_corr_{ft}_{unet}.pkl'
            if max_corr_file.is_file():
                with open(max_corr_file, 'rb') as f:
                    u_max[ustream] = pickle.load(f)
                    u_rsa_labels = u_max[ustream][2]
                    assert sem_labels == u_rsa_labels
            else:
                sem_idxs = []
                u_idxs = []
                for l in layers[unet]:
                    
                    uf = project_path / f'results/rsa/distances/{unet}_{ustream}_{l}.pkl'
                    with open(uf, 'rb') as f:
                        u_dist, u_labels = pickle.load(f)
                    if (sem_labels != u_labels) & (sem_idxs == []) & (u_idxs == []):
                        u_rsa_labels = sorted(list(set(sem_labels).intersection(u_labels)))
                        sem_idxs = [sem_labels.index(l) for l in u_rsa_labels]
                        u_idxs = [u_labels.index(l) for l in u_rsa_labels]
                        sem_dist = sem_dist[sem_idxs]
                        u_dist = u_dist[u_idxs]
                    elif (sem_labels != u_labels) & (sem_idxs != []) & (u_idxs != []):
                        u_dist = u_dist[u_idxs]
                    else:
                        u_rsa_labels = sem_labels
                    
                    assert len(sem_dist) == len(u_dist)
                    rsa, _ = spearmanr(sem_dist, u_dist)
                    
                    try:
                        if rsa > u_max[ustream][1]:
                            u_max[ustream] = [l, rsa, u_rsa_labels, u_dist]
                    except:
                        u_max[ustream] = [l, rsa, u_rsa_labels, u_dist]
                with open(max_corr_file, 'wb') as f:
                    pickle.dump(u_max[ustream], f)
        
        l_variance = {}
        for l in layers[f'{model}_{stream}']:
            if stream == 'img':
                file = f'{model}_{stream}_{l}.pkl'
            else:
                file = f'{model}_{stream}_concepts_{l}.pkl'
            with open(project_path / 'results/rsa/distances/' / file, 'rb') as f:
                c_dist, c_labels = pickle.load(f)
            
            sem_idx = []
            txt_idx = []
            img_idx = []
            if  (c_labels != u_rsa_labels) & (sem_idx == []):
                rsa_labels = sorted(list(set(c_labels).intersection(u_rsa_labels)))
                if rsa_labels != u_rsa_labels:
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
                    c_idx = [c_labels.index(l) for l in rsa_labels]
                    c_d = c_dist[c_idx]
            else:
                sem_d = sem_dist
                img_d = u_max['img'][-1]
                txt_d = u_max['txt'][-1]
                c_d = c_dist
                    
            X = np.column_stack((c_d, img_d, txt_d))
            l_variance[l] = get_r2(X, sem_d, [0,1,2]) - get_r2(X, sem_d, [1,2])
            print(f'{model} {stream} {ft} {l}: {l_variance[l]}')
        ft_variance[ft] = pd.DataFrame.from_dict(l_variance, orient='index')
    ft_variance = pd.concat(ft_variance)
    ft_variance.to_csv(project_path / 'results/rsa' / f'hs_var_{model}_{stream}.csv')


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
    parser.add_argument('-s', action='store', required=True)
    p_help = 'Path to the project folder containing the dataset and source code'
    parser.add_argument('-pp', action='store', required=True, help=p_help)

    model = parser.parse_args().m
    stream =  parser.parse_args().s
    project_path = Path(parser.parse_args().pp)

    compute_hs_similarity(project_path, model, stream)