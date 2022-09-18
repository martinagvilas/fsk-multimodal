import argparse
from pathlib import Path
import pickle

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from fsk.config import layers, multi_models
from fsk.dataprep.utils import get_synsets_ids
from fsk.it_match.load import (
    get_match, load_img_net_ft, load_multi_net_ft, load_txt_net_ft
)


def compute(project_path, model, stream, txt_type='concept'):
    
    # Define hidden states types
    if txt_type == 'concept':
        ht = 'concepts'
    elif txt_type == 'feature':
        ht = 'sem_features'
    
    # Get model information
    res_path = project_path / 'results' 
    m_path = res_path / model

    # Get image IDs
    s_imgs, _ = get_synsets_ids(project_path / 'dataset')
    imgs_ids = [i for s in s_imgs.values() for i in s]
    
    # Retrieve matching data 
    y = get_match(m_path, imgs_ids, txt_type=txt_type)

    # Compute mutual info
    for l_idx, l in enumerate(layers[f'{model}_{stream}']):
        if stream == 'img':
            file = res_path / 'mutual_info' / f'{model}_img_{l}.pkl'
            X = load_img_net_ft(project_path, model, l_idx)
            compute_mi(X, y, file)
        elif stream == 'txt':
            file = res_path / 'mutual_info' / f'{model}_txt_{l}_{ht}.pkl'
            X = load_txt_net_ft(project_path, model, l_idx, ht)
            compute_mi(X, y, file)
        elif stream == 'multi':
            file = res_path / 'mutual_info' / f'{model}_multi_{l}_{ht}.pkl'
            X = load_multi_net_ft(project_path, model, l_idx, ht)
            compute_mi(X, y, file)
                

def compute_mi(X, y, file):
    mi = []
    for i in range(y.shape[0]):
        i_mi = mutual_info_regression(X, y[i])
        mi.append(i_mi)
    mi = np.stack(mi)
    np.save(file, mi)


if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    m_help = 'Select which model to run. Can be one of the following options: \
        all, clip, vilt, albef'
    parser.add_argument('-m', action='store', required=True, help=m_help)
    s_help = 'Select which stream to run'
    parser.add_argument('-s', action='store', required=True, help=s_help)
    t_help = 'Select the type of text to run. Can be one of the following \
        options: concept, feature'
    parser.add_argument('-tt', action='store', required=True, help=t_help)
    p_help = 'Path to the project folder containing the dataset and source code'
    parser.add_argument('-pp', action='store', required=True, help=p_help)

    model = parser.parse_args().m
    stream = parser.parse_args().s
    txt_type = parser.parse_args().tt
    project_path = Path(parser.parse_args().pp)

    compute(project_path, model, stream, txt_type)