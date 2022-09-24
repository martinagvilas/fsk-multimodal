import argparse
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
import pandas as pd

from fsk.dataprep.utils import get_fsk_features, get_synsets_ids
from fsk.it_match.load import get_match


def compute_mutual_information(project_path, model):
    res_path = project_path / 'results'
    mi_path = res_path / 'mutual_info'
    mi_path.mkdir(parents=True, exist_ok=True)
    
    synsets_imgs, _ = get_synsets_ids(project_path / 'dataset')
    synsets = list(synsets_imgs.keys())
    fsk_ft = get_fsk_features(project_path / 'dataset')
    
    c_match, _ = get_match(res_path, model, synsets_imgs, center=False)
    c_match = MinMaxScaler(feature_range=(0.00001, 1)).fit_transform(c_match)
    ft_match, _ = get_match(
        res_path, model, synsets_imgs, m_type='feature', center=False
    )
    ft_match = MinMaxScaler(feature_range=(0.00001, 1)).fit_transform(ft_match)
    
    mi = []
    for c in range(c_match.shape[1]):
        mi.append(list(mutual_info_regression(ft_match, c_match[:,c])))
        print(f'done {c}', flush=True)
    
    mi = pd.DataFrame(mi, columns=fsk_ft, index=synsets)
    mi_file = mi_path / f'{model}.csv'
    mi.to_csv(mi_file)


if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    m_help = 'Select which model to run. Can be one of the following options: \
        all, clip, vilt, albef'
    parser.add_argument('-m', action='store', required=True, help=m_help)
    p_help = 'Path to the project folder containing the dataset and source code'
    parser.add_argument('-pp', action='store', required=True, help=p_help)

    model = parser.parse_args().m
    project_path = Path(parser.parse_args().pp)

    compute_mutual_information(project_path, model)