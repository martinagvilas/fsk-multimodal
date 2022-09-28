import argparse
from pathlib import Path

from fsk.similarity.rsa import RSA


COMPARISONS = [
    ['clip_txt', 'gpt', None], ['clip_img', 'vit_32', None],
    ['albef_txt', 'bert', None], ['albef_img', 'vit_16', None],
    ['albef_multi', 'bert', None], ['albef_multi', 'vit_16', None], 
    ['vilt_multi', 'bert', None], ['vilt_multi', 'vit_32', None]
]


def write_model_help(model):
    txt = (
        f'Select {model} to be compared with the RSA method. \
        Can be one of: semantic, albef_img, albef_txt, albef_multi, \
        clip_img, clip_txt, vilt_multi.'
    )
    return txt


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-all', action='store_true', required=False)
    parser.add_argument('-bi', action='store', required=False)
    m1_help = write_model_help('first model')
    parser.add_argument('-m1', action='store', required=False, help=m1_help)
    m2_help = write_model_help('second model')
    parser.add_argument('-m2', action='store', required=False, help=m2_help)
    p_help = 'Path to the project folder containing the dataset and source code'
    parser.add_argument('-pp', action='store', required=False, help=p_help)
    dm_help = ('Select a model to compute relative to predictions')
    parser.add_argument('-dm', action='store', required=False, help=dm_help)
    parser.add_argument('-pt', action='store', required=False, help=dm_help)
    
    model_1 = parser.parse_args().m1
    model_2 = parser.parse_args().m2
    project_path = Path(parser.parse_args().pp)
    decision_model = parser.parse_args().dm
    run_all = parser.parse_args().all
    batch_idx = int(parser.parse_args().bi)
    pred_type = parser.parse_args().pt

    if run_all:
        model_1, model_2, decision_model = COMPARISONS[batch_idx]
    RSA(project_path, model_1, model_2, decision_model, pred_type).compute()
