from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from scipy.io import loadmat

from fsk.dataprep.utils import get_synsets_ids

imagenet_meta = "/home/ts/Documents/ILSVRC2012_devkit_t12/data/meta.mat"
project_path = "/home/ts/PycharmProjects/fsk-multimodal"

meta_data = loadmat(imagenet_meta, struct_as_record=False)
synsets = np.squeeze(meta_data['synsets'])
ids = np.squeeze(np.array([x.ILSVRC2012_ID for x in synsets]))
wnids = np.squeeze(np.array([x.WNID for x in synsets]))
word = np.squeeze(np.array([x.words for x in synsets]))
num_children = np.squeeze(np.array([x.num_children for x in synsets]))
children = [x.children.astype(int).tolist()[0] for x in synsets]
df = pd.DataFrame(dict(id=ids, wnid=wnids, word=word, num_children=num_children, children=children))
df['wn_offset'] = df['wnid'].apply(lambda s: int(s[1:]))
df['synset'] = df['wnid'].apply(lambda s: wn.synset_from_pos_and_offset(s[0], int(s[1:])).name())
df = df.set_index('id')


def get_leaves(idx):
    row = df.loc[idx]
    if row.num_children == 0:
        return [idx]
    result = []
    for child in row.children:
        result.extend(get_leaves(child))
    return list(set(result))


df['leaves'] = df.index.map(get_leaves)

# map the network outcome idx to the wnid because the index is not the same
leaves = df[df.num_children == 0].sort_values('wn_offset').reset_index(drop=True)
net_idx_to_wnid = leaves.wnid.to_dict()
wnid_to_net_idx = {v: k for k, v in net_idx_to_wnid.items()}

synsets_ids, _ = get_synsets_ids(Path(project_path) / 'dataset')
things_synsets = list(map(lambda x: wn.synset('.'.join(x.split('-'))).name(), synsets_ids.keys()))

unknown_synset = []
wnid_to_things = defaultdict(list)

for synset in things_synsets:
    in_sysnet = df[df.synset == synset]  # get the corresponding synset in the in hierarchy
    if in_sysnet.shape[0] == 0:
        unknown_synset.append(synset)
        continue
    elif in_sysnet.shape[0] > 1:
        raise ValueError(synset)
    # match all leaves (classes that can be predicted by the network) to the concept
    leaves = in_sysnet.iloc[0].leaves
    for leave in leaves:
        wnid_to_things[df.loc[leave].wnid].append(synset)
print(f"Found {len(wnid_to_things)} synsets in the imagenet hierarchy. "
      f"{len(unknown_synset)} could not be assigned.")

net_idx_to_concept = {wnid_to_net_idx[k]: v for k, v in wnid_to_things.items()}
with open(Path(project_path) / 'results' / 'vit_to_concept.json', 'w') as f:
    json.dump(net_idx_to_concept, f, sort_keys=True, indent=4)

net_idx_to_synset = {k: df[df.wnid == v].iloc[0].synset for k, v in net_idx_to_wnid.items()}
with open(Path(project_path) / 'results' / 'vit_to_synset.json', 'w') as f:
    json.dump(net_idx_to_synset, f, sort_keys=True, indent=4)
