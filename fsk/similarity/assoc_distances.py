import argparse
import json
from pathlib import Path

import gensim.downloader as api
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import pandas as pd
import spacy
import scipy.sparse as sp

from fsk.dataprep.utils import get_synsets_ids


W_TYPES = ['NOUN', 'VERB', 'ADJ']


def compute_mi_similarity(project_path, model):
    synsets_imgs, concepts = get_synsets_ids(project_path / 'dataset')
    synset_map = {s: c for s, c in zip(synsets_imgs.keys(), concepts['things'])}
    mi = pd.read_csv(project_path / 'results/mutual_info' / f'{model}.csv')
    mi['concepts'] = [synset_map[s] for s in mi['Unnamed: 0']]
    mi = mi.drop(columns='Unnamed: 0').set_index('concepts')

    # Clean features
    nlp = spacy.load('en_core_web_sm')
    column_map = [clean_feature(ft, nlp) for ft in mi.columns] * 342
    mi = mi.stack().to_frame()
    mi['feature'] = column_map
    mi = mi.reset_index().drop(columns='level_1').rename(columns={0: 'mi'})

    # Compute similarity with wikipedia co-occurrence
    wiki_co_occur_f = project_path / 'dataset/co_occur' / 'co_occur_wiki.npz'
    wiki_co_occur = sp.load_npz(wiki_co_occur_f)
    wiki_words_f = project_path / 'dataset/co_occur' / 'features_wiki.json'
    with open(wiki_words_f, 'rb') as f:
        wiki_words = json.load(f)
    dist = []
    for _, row in mi.iterrows():
        try:
            w1 = row['concepts']
            w1_idx = wiki_words.index(w1)
            d_w1 = []
            for w2 in row['feature']:
                w2_idx = wiki_words.index(str(w2))
                d_w1.append(wiki_co_occur[w1_idx, w2_idx])
            dist.append(np.mean(d_w1))
        except:
            dist.append(np.nan)

    # Compute similarity with wikipedia glove representation

    # Compute similarity with vico co-occurrence 

    # Compute similarity with vico glove representation

def clean_feature(ft, nlp):
    ft = (' ').join(ft.split('_'))
    new_ft = [w for w in nlp(ft) if w.pos_ in W_TYPES]    
    return new_ft

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    m_help = 'Select which model to run. Can be one of the following options: \
        all, clip, vilt, albef'
    parser.add_argument('-m', action='store', required=True, help=m_help)
    p_help = 'Path to the project folder containing the dataset and source code'
    parser.add_argument('-pp', action='store', required=True, help=p_help)

    model = parser.parse_args().m
    project_path = Path(parser.parse_args().pp)

    compute_mi_similarity(project_path, model)