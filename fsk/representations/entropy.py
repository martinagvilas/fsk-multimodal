import argparse
from pathlib import Path

from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from fsk.dataprep.utils import get_fsk_features, get_synsets_ids
from fsk.it_match.load import get_match


def compute_KLD(project_path, model):
    res_path = project_path / 'results'
    kld_path = res_path / 'kld'
    res_path.mkdir(parents=True, exist_ok=True)
    
    synsets_imgs, _ = get_synsets_ids(project_path / 'dataset')
    synsets = list(synsets_imgs.keys())
    fsk_ft = get_fsk_features(project_path / 'dataset')
    
    c_match, _ = get_match(res_path, model, synsets_imgs, center=False)
    c_match = MinMaxScaler(feature_range=(0.00001, 1)).fit_transform(c_match)
    ft_match, _ = get_match(
        res_path, model, synsets_imgs, m_type='feature', center=False
    )
    ft_match = MinMaxScaler(feature_range=(0.00001, 1)).fit_transform(ft_match)
    
    ecf = []
    efc = []
    for c in range(c_match.shape[1]):
        for f in range(ft_match.shape[1]):
            ecf.append([synsets[c], fsk_ft[f], entropy(c_match[:,c], ft_match[:,f])])
            efc.append([fsk_ft[f], synsets[c], entropy(ft_match[:,f], c_match[:,c])])
    
    ecf_info = pd.DataFrame(ecf, columns=['synset', 'feature', 'entropy'])
    ecf_file = kld_path / f'{model}_entropy_concept_feature.csv'
    ecf_info.to_csv(ecf_file)

    efc_info = pd.DataFrame(efc, columns=['feature', 'synset', 'entropy'])
    efc_file = kld_path / f'{model}_entropy_feature_concept.csv'
    efc_info.to_csv(efc_file)


if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    m_help = 'Select which model to run. Can be one of the following options: \
        all, clip, vilt, albef'
    parser.add_argument('-m', action='store', required=True, help=m_help)
    p_help = 'Path to the project folder containing the dataset and source code'
    parser.add_argument('-pp', action='store', required=True, help=p_help)

    model = parser.parse_args().m
    project_path = Path(parser.parse_args().pp)

    compute_KLD(project_path, model)