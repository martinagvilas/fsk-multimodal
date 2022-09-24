import argparse
from pathlib import Path
import pickle

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from fsk.config import layers, multi_models
from fsk.dataprep.utils import get_fsk_features, get_synsets_ids
from fsk.it_match.load import (
    get_match, load_img_net_ft, load_multi_net_ft, load_txt_net_ft
)


def compute(project_path, model, stream, layer,  txt_type):
    res_path = project_path / 'results'
    net_ft_path = res_path / model / 'net_ft'
    preds_path = project_path / 'results' / 'regression'
    preds_path.mkdir(parents=True, exist_ok=True)
    synsets_imgs, _ = get_synsets_ids(project_path / 'dataset')
    model_name = f'{model}_{stream}'
    l_idx = layers[model_name].index(layer)

    if txt_type:
        features = get_fsk_features(project_path / 'dataset')

    preds, preds_labels = get_match(res_path, model, synsets_imgs, m_type='feature')
    
    if stream == 'img':
        X, X_labels = load_img_net_ft(net_ft_path, layer, l_idx, synsets_imgs)
        assert preds_labels == X_labels
    elif stream == 'txt':
        X = load_txt_net_ft(net_ft_path, layer, l_idx, 'concept')
        s_lens = [len(s_imgs) for s, s_imgs in synsets_imgs.items()]
        X = np.repeat(X, s_lens, axis=0)
    elif stream == 'multi':
        X, X_labels = load_multi_net_ft(net_ft_path, layer, l_idx, synsets_imgs)
        assert preds_labels == X_labels
    print('Loaded X', flush=True)

    for f_idx in range(preds.shape[1]):
        res = {}
        # Compute mutual information
        res['mutual_information'] = mutual_info_regression(X, preds[:, f_idx])

        # Compute linear regression
        X_train, X_test, y_train, y_test = train_test_split(X, preds[:, f_idx])
        reg = Ridge()
        reg.fit(X_train, y_train)
        true_score = r2_score(y_test, reg.predict(X_test))

        # random_scores = []
        # for _ in range(100):
        #     np.random.shuffle(y_train)
        #     reg.fit(X_train, y_train)
        #     random_scores.append(mean_squared_error(y_test, reg.predict(X_test)))

        # pval = np.sum(random_scores < true_score) / 100

        res['preds'] = reg.predict(X)
        res['r2'] = true_score
        #res['pval'] = pval
        res['coefs'] = reg.coef_
        res['intercept'] = reg.intercept_

        # reg_preds = []
        # scores = []
        # coefs = []
        # intercept = []
        # for train_index, test_index in kf.split(X):
        #     X_train = X[train_index]
        #     X_test = X[test_index]
        #     y_train = preds[train_index, f_idx]
        #     y_test = preds[test_index, f_idx]
        #     reg = Ridge()
        #     reg.fit(X_train, y_train)
        #     reg_preds.append(reg.predict(X))
        #     scores.append(mean_squared_error(y_test, reg.predict(X_test)))
        #     coefs.append(reg.coef_)
        #     intercept.append(reg.intercept_)

        # Save results
        # res = {}
        # res['preds'] = np.mean(reg_preds, axis=0)
        # res['mse'] = np.mean(scores)
        # res['coefs'] = np.mean(coefs, axis=0)
        # res['intercept'] = np.mean(intercept)
        # res['mutual_information'] = mi

        print(
            f'feature {features[f_idx]}, {model} {stream} {layer} score: '\
            f'{res["r2"]}'
        )
        
        res_file = preds_path / f'{model}_{stream}_{layer}_{txt_type}_{f_idx}.pkl'
        with open(res_file, 'wb') as f:
            pickle.dump(res, f)
        

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('-all', action='store_true', required=False)
    parser.add_argument('-bi', action='store', required=False)
    parser.add_argument('-m', action='store', required=False)
    parser.add_argument('-s', action='store', required=False)
    parser.add_argument('-l', action='store', required=False)
    parser.add_argument('-tt', action='store', required=False)
    parser.add_argument('-pp', action='store', required=True)

    project_path = Path(parser.parse_args().pp)

    all = parser.parse_args().all
    if all != None:
        batch_idx = int(parser.parse_args().bi)
        model, stream, layer = multi_models[batch_idx]

    else:
        model = parser.parse_args().m
        stream = parser.parse_args().s
        layer = parser.parse_args().l
        txt_type = parser.parse_args().tt
    
    compute(project_path, model, stream, layer, 'feature')