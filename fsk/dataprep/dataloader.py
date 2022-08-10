from pathlib import Path
from PIL import Image

import pandas as pd
from torch.utils.data import Dataset

from fsk.dataprep.things import load_things_imgs_dir
from fsk.dataprep.utils import get_concepts_info, get_fsk_features


class ThingsDataset(Dataset):
    def __init__(self, dataset_path, batch_idx=None, batch_size=75):
        self.dataset_path = Path(dataset_path)
        self.things_path = self.dataset_path / 'things'
        self.annot_path = self.dataset_path / 'annotations'
        self.stimuli_info = self.get_stimuli_info()

    def get_stimuli_info(self):
        concepts_info = get_concepts_info(self.dataset_path)
        imgs_dirs = load_things_imgs_dir(
            self.things_path, concepts_info['ids_things'].tolist()
        )
        stim_info = []
        for _, row in concepts_info.iterrows():
            concept_id = row['ids_things']
            for i_dir in imgs_dirs[concept_id]:
                new_row = row.copy()
                new_row['img_path'] = i_dir
                stim_info.append(new_row)
        stim_info = pd.concat(stim_info, axis=1).T
        return stim_info

    def __len__(self):
        return len(self.stimuli_info)

    def __getitem__(self, idx):
        out = {}
        item_info = self.stimuli_info.iloc[idx]

        out['img_id'] = item_info['img_path'].stem
        out['img'] = Image.open(item_info['img_path']).convert('RGB')

        out['concepts_things'] = item_info['concepts_things']
        out['concepts_mcrae'] = item_info['concepts_mcrae']
        out['synset'] = item_info['synsets'] 
        out['features'] = get_fsk_features(self.dataset_path, [out['synset']])

        return out
