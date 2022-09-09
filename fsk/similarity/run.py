import argparse
from pathlib import Path

from fsk.similarity.rsa import RSA


def write_model_help(model):
    txt = (
        f'Select {model} to be compared with the RSA method. \
        Can be one of: semantic, albef_img, albef_txt, albef_multi, \
        clip_img, clip_txt, vilt_multi.'
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
    ft_help = (
        'Select feature type. Can be on of: taxonomic, encyclopaedic, '
        'function, visual_perceptual, other_perceptual.'
    )
    parser.add_argument('-ft', action='store', required=False, help=p_help)
    
    model_1 = parser.parse_args().m1
    model_2 = parser.parse_args().m2
    project_path = Path(parser.parse_args().pp)
    ft = parser.parse_args().ft

    rsa = RSA(project_path, model_1, model_2, ft).compute()
    print('done')