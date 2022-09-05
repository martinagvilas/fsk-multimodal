import argparse
from pathlib import Path

from fsk.human_similarity.rsa import RSA


def write_model_help(model):
    txt = (
        f'Select {model} to be compared with the RSA method. \
        Can be one of: semantic, albef_visual, albef_text, albef_multimodal, \
        clip_visual, clip_text, vilt_multimodal.'
    )
    return txt


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    m1_help = write_model_help('first model')
    parser.add_argument('-m1', action='store', required=True, help=m1_help)
    m2_help = write_model_help('second model')
    parser.add_argument('-m2', action='store', required=True, help=m2_help)
    p_help = 'Path to the project folder containing the dataset and source code'
    parser.add_argument('-pp', action='store', required=True, help=p_help)
    
    model_1 = parser.parse_args().m1
    model_2 = parser.parse_args().m2
    project_path = Path(parser.parse_args().pp)

    rsa = RSA(project_path, model_1, model_2)