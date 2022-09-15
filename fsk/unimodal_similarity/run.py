import argparse
from pathlib import Path

import torch
from fsk.unimodal_similarity.rsa import RSA


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    m_help = 'Select which model to run. Can be one of the following options: \
        clip, vilt, albef'
    parser.add_argument('-m', action='store', required=True, help=m_help)
    p_help = 'Path to the project folder containing the dataset and source code'
    parser.add_argument('-pp', action='store', required=True, help=p_help)
    model = parser.parse_args().m
    project_path = Path(parser.parse_args().pp)

    RSA(project_path, model).compute()