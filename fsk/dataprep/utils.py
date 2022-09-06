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
        concepts_info = get_concepts_info(dataset_path)
        mcrae_info = load_mcrae_info(annot_path)
        fsk = mcrae_info[['Concept', 'Feature', 'BR_Label']]
        # Map concepts to synsets
        mcrae_synsets = load_mcrae_synsets_info(annot_path)
        synset_map = pd.Series(
            mcrae_synsets['Synset'].values, index=mcrae_synsets['Concept']
        ).to_dict()
        fsk = fsk.replace(to_replace=synset_map)
        # Filter synsets
        fsk = fsk.loc[fsk['Concept'].isin(concepts_info['synsets'])]
        fsk.columns = ['synsets', 'features', 'features_type']
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
        # Save
        fsk.to_csv(file, index=False)
    return fsk


def get_concepts_info(dataset_path, overwrite=False):
    """Get info about shared concepts in the Things and McRae dataset.

    Parameters
    ----------
    dataset_path : str or pathlib Path
        Path to folder containing dataset information.

    Returns
    -------
    pd DataFrame
        Info about shared concepts in the Things and McRae dataset.
    """
    ## TODO: which info?
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


def get_fsk_features(dataset_path, synsets=None):
    fsk = get_fsk(dataset_path)
    if synsets != None:
        fsk = fsk.loc[fsk['synsets'].isin(synsets)]
    features = fsk['features'].unique().tolist()
    return features


def get_fsk_synsets(dataset_path):
    fsk = get_fsk(dataset_path)
    synsets = fsk['synsets'].unique().tolist()
    return synsets


def get_concepts(dataset_path, source='things'):
    """Get list of shared shared concepts in the Things and McRae dataset.

    Parameters
    ----------
    dataset_path : str or pathlib Path
        Path to folder containing dataset information.

    Returns
    -------
    list
       Shared concepts in the Things and McRae dataset.
    """
    concepts_info = get_concepts_info(dataset_path)
    concepts = concepts_info[f'concepts_{source}'].tolist()
    return concepts


def get_synsets_ids(dataset_path):
    imgs_path = dataset_path / 'things/object_images'
    concepts_info = get_concepts_info(dataset_path)
    imgs_ids = {}
    concepts = {'mcrae': [], 'things': []}
    for _, row in concepts_info.iterrows():
        synset = row['synsets']
        concept_dir = imgs_path / row['ids_things']
        concepts['mcrae'].append(row['concepts_mcrae'])
        concepts['things'].append(row['concepts_things'])
        concept_imgs = [
            d.stem for d in concept_dir.iterdir() if d.suffix == '.jpg'
        ]
        imgs_ids[synset] = concept_imgs
    return imgs_ids, concepts