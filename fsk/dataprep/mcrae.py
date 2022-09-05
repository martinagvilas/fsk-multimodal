from pathlib import Path

import numpy as np
import pandas as pd


def load_mcrae_info(annot_path):
    file = Path(annot_path) / 'mcrae_info.txt'
    mcrae_info = pd.read_csv(file, sep="\t")
    return mcrae_info


def load_mcrae_synsets_info(annot_path):
    file = Path(annot_path) / 'mcrae_synsets_annotated.csv'
    synsets_info = pd.read_csv(file, sep=';')
    synsets_info =synsets_info.rename(columns={
        'label': 'Concept', 'synset': 'Synset'
    })
    synsets_info['Synset'] = synsets_info['Synset'].replace(
        regex=r"\.", value='-'
    )
    return synsets_info


# def get_sem_feature_matrix(annot_path):
#     mcrae_file = annot_path / 'mcrae_features_info.txt'
#     mcrae_info = pd.read_csv(mcrae_file, sep="\t")
#     mcrae_info['Feature'] = (
#         mcrae_info['Feature'].replace(to_replace=r".+_-_", value='', regex=True)
#     )
#     mcrae_info['Concept'] = mcrae_info['Concept'].replace(to_replace={
#         'bin_(waste_)': 'wastebin', 'board_(black)': 'blackboard',
#         'hose_(leggings)': 'leggings', 'drapes': 'curtains', 'nylons': 'nylon', 
#         'onions': 'onion', 'pajamas': 'pajama'
#     })
#     f_matrix = pd.DataFrame(
#         np.zeros((len(concepts), len(features))), 
#         index=concepts, columns=features
#     )
#     for _, row in mcrae_info.iterrows():
#         if row['Concept'] in concepts:
#             f_matrix.loc[row['Concept'], row['Feature']] = row['Prod_Freq']
#     return f_matrix
