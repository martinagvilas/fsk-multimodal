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
