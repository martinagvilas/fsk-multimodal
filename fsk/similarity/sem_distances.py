from itertools import combinations
import pickle

import numpy as  np
import pandas as pd
from scipy.spatial.distance import pdist

from fsk.dataprep.mcrae import load_mcrae_info
from fsk.dataprep.utils import get_fsk_features


def get_mcrae_distances(project_path, synsets, concepts, overwrite=False):
    dist_file = project_path / 'results/rsa/distances/mcrae.pkl'
    if (dist_file.is_file()) and (overwrite == False):
        with open(dist_file, 'rb') as f:
            dist, labels = pickle.load(f)
    else:
        sem_matrix = get_mcrae_feature_matrix(project_path, synsets, concepts)
        dist = pdist(sem_matrix.values, metric='cosine')
        labels = list(combinations(sem_matrix.index.tolist(), 2))
        assert len(labels) == len(dist)
        with open(dist_file, 'wb') as f:
            pickle.dump((dist, labels), f)

    return (dist, labels)


def get_mcrae_feature_matrix(project_path, synsets, concepts, overwrite=False):
    feature_matrix_file = project_path/ 'dataset/annotations/mcrae_features.csv'
    if (feature_matrix_file.is_file()) and (overwrite == False):
        f_matrix = pd.read_csv(feature_matrix_file, index_col=0)
    else:
        # Load mcrae feature information
        mcrae_info = load_mcrae_info(project_path/ 'dataset/annotations')
        # Load features of interest
        features = get_fsk_features(project_path / 'dataset')
        # Create feature matrix
        f_matrix =  pd.DataFrame(
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
                f_matrix.loc[synset, feature] = row['Prod_Freq']
        # Save
        f_matrix.to_csv(feature_matrix_file)
    return f_matrix



