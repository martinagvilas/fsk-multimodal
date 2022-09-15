from pathlib import Path
from PIL import Image

import pandas as pd
from torch.utils.data import Dataset

from fsk.dataprep.things import load_things_imgs_dir
from fsk.dataprep.utils import get_concepts_info, get_fsk_features


class ThingsDataset(Dataset):
    def __init__(self, dataset_path, batch_idx=None):
        self.dataset_path = Path(dataset_path)
        self.things_path = self.dataset_path / 'things'
        self.annot_path = self.dataset_path / 'annotations'
        self.stimuli_info = self.get_stimuli_info(batch_idx)

    def get_stimuli_info(self, batch_idx):
        concepts_info = get_concepts_info(self.dataset_path)
        imgs_dirs = load_things_imgs_dir(
            self.things_path, concepts_info['ids_things'].tolist()
        )
        if batch_idx != None:
            imgs_dirs = get_batch_imgs(imgs_dirs, batch_idx)
        stim_info = []
        for _, row in concepts_info.iterrows():
            concept_id = row['ids_things']
            try:
                for i_dir in imgs_dirs[concept_id]:
                    new_row = row.copy()
                    new_row['img_path'] = i_dir
                    stim_info.append(new_row)
            except KeyError:
                continue
        stim_info = pd.concat(stim_info, axis=1).T
        return stim_info

    def __len__(self):
        return len(self.stimuli_info)

    def __getitem__(self, idx):
        out = {}
        item_info = self.stimuli_info.iloc[idx]

        out['img_id'] = item_info['img_path'].stem
        out['img'] = Image.open(item_info['img_path']).convert('RGB')

        out['concepts_things'] = [item_info['concepts_things']]
        out['concepts_mcrae'] = [item_info['concepts_mcrae']]
        out['synset'] = [item_info['synsets']]
        out['features'] = get_fsk_features(self.dataset_path, out['synset'])

        return out


def get_batch_imgs(imgs_ids, batch_id, batch_size=50):
    batch_id = int(batch_id)
    batch_start = batch_id * batch_size
    batch_end = batch_start + batch_size
    if len(imgs_ids) < batch_end:
        batch_end = len(imgs_ids)
    k_ids = sorted(list(imgs_ids.keys()))[batch_start:batch_end]
    sel_imgs_ids = {k: imgs_ids[k]  for k in k_ids}
    return sel_imgs_ids