from itertools import combinations

import torch


def get_concept_match_average(c_match_path, synset_ids):
    #labels = list(combinations(list(synset_ids.keys()), 2))
    synsets = list(synset_ids.keys())
    labels = []
    preds = []
    for idx, s1 in enumerate(synsets):
        s_pred = []
        for img in synset_ids[s1]:
            s_pred.append(torch.load(
                c_match_path / f'{img}.pt', map_location=torch.device('cpu')
            ))
        s_pred = torch.mean(torch.stack(s_pred), dim=0)
        for s2 in range(idx+1, len(synsets)):
            preds.append(s_pred[s2])
            labels.append((s1, synsets[s2]))
    preds = 1 - torch.stack(preds).detach().numpy()
    return preds, labels


def get_concept_match_accuracy(c_match_path, synsets, s_idxs, top_k):
    m_preds = []
    for s, s_imgs in s_idxs.items():
        correct_val = synsets.index(s)
        for img in s_imgs:
            val = [s, img]
            i_pred = torch.load(
                c_match_path / f'{img}.pt', map_location=torch.device('cpu')
            )
            top_indices = i_pred.topk(15)[1]
            val.append([synsets[i] for i in top_indices])
            for t in top_k:
                val.append(correct_val in top_indices[:t])
            m_preds.append(val)
