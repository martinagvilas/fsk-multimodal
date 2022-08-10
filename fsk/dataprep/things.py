from pathlib import Path

import pandas as pd


def load_things_info(things_path):
    things_file = Path(things_path) / 'things_concepts.tsv'
    things_info = pd.read_csv(things_file, sep='\t')
    return things_info


def load_things_synsets_info(things_path):
    things_info = load_things_info(Path(things_path))
    synsets_info = things_info[['Word', 'uniqueID', 'Wordnet ID4']]
    synsets_info = synsets_info.rename(
        columns={'Word': 'Concept', 'uniqueID': 'ID', 'Wordnet ID4': 'Synset'}
    )
    synsets_info['Synset'] = synsets_info['Synset'].replace(regex=r"\.", value='-')
    return synsets_info


def load_things_imgs_dir(things_path, concepts=None):
    imgs_path = Path(things_path) / 'object_images'
    imgs_paths = {}
    for c_dir in imgs_path.iterdir():
        concept_name = c_dir.name
        if concept_name in concepts:
            concept_dirs = [d for d in c_dir.iterdir() if d.suffix == '.jpg']
            imgs_paths[concept_name] = concept_dirs
        else:
            continue
    return imgs_paths
