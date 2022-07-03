from collections import OrderedDict
import json
from pathlib import Path

import pandas as pd

from fsk.dataprep.things import load_things_synsets_info
from fsk.dataprep.mcrae import load_mcrae_info, load_mcrae_synsets_info


def get_fsk(dataset_path, overwrite=False):
    dataset_path = Path(dataset_path)
    annot_path = dataset_path / 'annotations'
    file = annot_path / 'fsk_info.csv'
    if (file.is_file()) and overwrite == False:
        fsk = pd.read_csv(file)
    else:
        concepts_info = get_common_concepts(dataset_path)
        concepts_mcrae = [
            c.split(', ') if ',' in c else [c] 
            for c in concepts_info['concepts_mcrae']
        ]
        concepts_mcrae = [i for c in concepts_mcrae for i in c]
        mcrae_info = load_mcrae_info(annot_path)
        fsk = mcrae_info[['Concept', 'Feature', 'BR_Label']]
        fsk.columns = ['concepts_mcrae', 'features', 'features_type']
        fsk = fsk.loc[fsk['concepts_mcrae'].isin(concepts_mcrae)]
        # Replace feature type names
        fsk['features_type'] = fsk['features_type'].replace(
            regex=r"^visual.+", value='visual_perceptual'
        )
        fsk['features_type'] = fsk['features_type'].replace(
            regex=['sound', 'tactile', 'smell', 'taste'], 
            value='other_perceptual'
        )
        # Replace dataset-specific symbols
        fsk['features'] = fsk['features'].replace(regex=r".+_-_", value='')
        # Add idx to features
        f_map = get_feature_idx_map(annot_path, fsk['features'].unique())
        fsk['features_idx'] = fsk['features'].map(f_map)
        # Merge with other concepts data
        for idx, r_info in fsk.iterrows():
            c_info = concepts_info.loc[
                concepts_info['concepts_mcrae']
                .str.contains(r_info['concepts_mcrae'])
            ]
            fsk.loc[idx,'concepts_things'] = c_info['concepts_things'].values[0]
            fsk.loc[idx,'synsets']= c_info['synsets'].values[0]
        # Reorder data and save
        fsk = fsk[[
            'concepts_things', 'concepts_mcrae', 'synsets', 
            'features', 'features_idx', 'features_type'
        ]]
        fsk.to_csv(file, index=False)
    return fsk


def get_concepts_info(dataset_path, overwrite=False):
    """Get info about shared concepts in the Things and McRae dataset.

    Parameters
    ----------
    dataset_path : str or pathlib Path
        Path to dataset.

    Returns
    -------
    pd DataFrame
        Info about shared concepts in the Things and McRae dataset.
    """
    annot_path = Path(dataset_path) / 'annotations'
    file = annot_path / 'concepts_info.csv'
    if file.is_file() and overwrite == False:
        c_info = pd.read_csv(file)
    else:
        things_info = load_things_synsets_info(Path(dataset_path) / 'things')
        mcrae_info = load_mcrae_synsets_info(annot_path)
        common_synsets = [
            s for s in things_info['Synset'] 
            if (s in mcrae_info['Synset'].tolist()) & (isinstance(s, str))
        ]
        print(f'Found {len(common_synsets)} common synsets', flush=True)
        things_concepts = [
            things_info.loc[things_info['Synset']==s]['Concept'].values[0]
            for s in common_synsets
        ]
        things_ids = [
            things_info.loc[things_info['Synset']==s]['ID'].values[0]
            for s in common_synsets
        ]
        mcrae_concepts = [
            (', ').join(
                mcrae_info.loc[mcrae_info['Synset']==s]['Concept'].tolist()
            ) for s in common_synsets
        ]
        c_info = pd.DataFrame({
            'concepts_things': things_concepts, 'ids_things': things_ids, 
            'concepts_mcrae': mcrae_concepts, 'synsets': common_synsets
        })
        c_info['concepts_mcrae'] = c_info['concepts_mcrae'].replace(
            regex=r"_.+", value=''
        )
        c_info.to_csv(file, index=False)
    return c_info


def get_feature_idx_map(annot_path, features=None):
    f_file = Path(annot_path) / 'feature_idx_map.json'
    if features is not None:
        features_map = OrderedDict({f:i for i, f in enumerate(features)})
        with open(f_file, 'w') as f:
            json.dump(features_map, f)
    else:
        with open(f_file, 'r') as f:
            features_map = json.load(f, object_pairs_hook=OrderedDict)
    return features_map


def get_fsk_features(dataset_path, concepts=None):
    fsk = get_fsk(dataset_path)
    if concepts != None:
        fsk = fsk.loc[fsk['concepts_things'].isin(concepts)]
    features = fsk['features'].unique().tolist()
    return features


def get_fsk_concepts(dataset_path, source='things'):
    fsk = get_fsk(dataset_path)
    if source == 'things':
        concepts = fsk['concepts_things'].unique().tolist()
    elif source == 'mcrae':
        print("Not implemented yet")
    return concepts


def get_fsk_concepts_ids(dataset_path):
    fsk = get_fsk(dataset_path)
    concepts_ids = fsk['concepts_ids'].unique().tolist()
    return concepts_ids