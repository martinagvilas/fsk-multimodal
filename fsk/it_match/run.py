import argparse
from pathlib import Path

import torch
from fsk.it_match import itm_clip, itm_albef, itm_vilt


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', action='store', required=True)
    parser.add_argument('-pp', action='store', required=True)
    model = parser.parse_args().m
    project_path = Path(parser.parse_args().pp)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model == 'all':
        itm_clip.run(project_path, device)
        itm_vilt.run(project_path, device)
        itm_albef.run(project_path, device)
    elif model == 'clip':
        itm_clip.run(project_path, device)
    elif model == 'vilt':
        itm_vilt.run(project_path, device)
    elif model == 'albef':
        itm_albef.run(project_path, device)