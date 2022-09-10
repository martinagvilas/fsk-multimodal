from itertools import combinations
import pickle

import numpy as  np
import pandas as pd
from scipy.spatial.distance import pdist

from fsk.dataprep.mcrae import load_mcrae_info
from fsk.dataprep.utils import get_fsk_features


def get_mcrae_distances(
    project_path, synsets, concepts, ft_type, overwrite=False
):
    res_path = project_path / 'results/rsa/distances/'
    res_path.mkdir(parents=True, exist_ok=True)

    if ft_type == None:
        dist_file = res_path / 'mcrae.pkl'
    else:
        dist_file = res_path / f'mcrae_{ft_type}.pkl'

    if (dist_file.is_file()) and (overwrite == False):
        with open(dist_file, 'rb') as f:
            dist, labels = pickle.load(f)
    else:
        ft = get_mcrae_features(project_path, synsets, concepts, ft_type)
        dist = pdist(ft.values, metric='cosine')
        labels = list(combinations(ft.index.tolist(), 2))
        assert len(labels) == len(dist)
        with open(dist_file, 'wb') as f:
            pickle.dump((dist, labels), f)

    dist_dict = {'sem_mcrae': dist}
    return (dist_dict, labels)


def get_mcrae_features(
    project_path, synsets, concepts, ft_type, overwrite=False
):
    if ft_type == None:
        file = project_path/ 'dataset/annotations/mcrae_features.csv'
    else:
        file = project_path/ f'dataset/annotations/mcrae_features_{ft_type}.csv'
    
    if (file.is_file()) and (overwrite == False):
        ft = pd.read_csv(file, index_col=0)
    else:
        # Load mcrae features information
        dataset_path = project_path / 'dataset'
        mcrae_info = load_mcrae_info(dataset_path/ 'annotations')
        features = get_fsk_features(dataset_path, filter=ft_type)
        # Create feature matrix
        ft =  pd.DataFrame(
            np.zeros((len(synsets), len(features))), 
            index=synsets, columns=features
        )
        for concept, synset in zip(concepts, synsets):
            c_names = concept.split(', ')
            c_info = mcrae_info.loc[mcrae_info['Concept'].isin(c_names)]
            c_info['Feature'] = c_info['Feature'].replace(
                regex=r".+_-_", value=''
            )
            for _, row in c_info.iterrows():
                feature = row['Feature']
                if feature in features:
                    ft.loc[synset, feature] = row['Prod_Freq']
        # Remove rows with all zeros
        ft = ft.loc[~(ft==0).all(axis=1)]
        # Save
        ft.to_csv(file)
    return ft



