import argparse
from pathlib import Path

import torch
from fsk.it_match import itm_clip, itm_albef, itm_vilt


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    m_help = 'Select which model to run. Can be one of the following options: \
        all, clip, vilt, albef'
    parser.add_argument('-m', action='store', required=True, help=m_help)
    p_help = 'Path to the project folder containing the dataset and source code'
    parser.add_argument('-pp', action='store', required=True, help=p_help)
    model = parser.parse_args().m
    project_path = Path(parser.parse_args().pp)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model == 'all':
        itm_clip.compute(project_path, device)
        itm_vilt.compute(project_path, device)
        itm_albef.compute(project_path, device)
    elif model == 'clip':
        itm_clip.compute(project_path, device)
    elif model == 'vilt':
        itm_vilt.compute(project_path, device)
    elif model == 'albef':
        itm_albef.compute(project_path, device)