import argparse
from pathlib import Path

import torch

from fsk.unet_features.language import FtBert, FtGPT
from fsk.unet_features.vit import FtVit


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    m_help = 'Select which model to run. Can be one of the following options: \
        vit_16, vit_32, bert, gpt'
    parser.add_argument('-m', action='store', required=True, help=m_help)
    p_help = 'Path to the project folder containing the dataset and source code'
    parser.add_argument('-pp', action='store', required=True, help=p_help)
    model = parser.parse_args().m
    project_path = Path(parser.parse_args().pp)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model.startswith('vit'):
        version = model.split('_')[1]
        FtVit(version, project_path, device=device).compute()
    elif model == 'bert':
        FtBert(project_path, text_type='concept').compute()
    elif model == 'gpt':
        FtGPT(project_path, text_type='concept').compute()