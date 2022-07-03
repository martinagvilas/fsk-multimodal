from pathlib import Path

import pandas as pd


def load_mcrae_info(annot_path):
    file = Path(annot_path) / 'mcrae_info.txt'
    mcrae_info = pd.read_csv(file, sep="\t")
    return mcrae_info


def load_mcrae_synsets_info(annot_path):
    file = Path(annot_path) / 'mcrae_synsets_annotated.csv'
    mcrae_synsets_info = pd.read_csv(file, sep=';')
    mcrae_synsets_info = mcrae_synsets_info.rename(columns={
        'label': 'Concept', 'synset': 'Synset'
    })
    return mcrae_synsets_info
