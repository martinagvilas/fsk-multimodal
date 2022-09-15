# Layer information
layers = {
    'albef_img': [f'hs_{n}' for n in range(12)] + ['c-out'],
    'albef_txt': [f'hs_{n}' for n in range(6)] + ['c-out'],
    'albef_multi': [f'hs_{n}' for n in range(6)],
    'clip_img': [f'hs_{n}' for n in range(12)] + ['c-out'],
    'clip_txt': [f'hs_{n}' for n in range(12)] + ['c-out'],
    'vilt_multi': [f'hs_{n}' for n in range(12)],
    'bert_concepts': [f'hs_{n}' for n in range(12)],
    'bert_definition': [f'hs_{n}' for n in range(12)],
    'gpt_concepts': [f'hs_{n}' for n in range(12)],
    'gpt_definition': [f'hs_{n}' for n in range(12)],
    'vit_16': [f'hs_{n}' for n in range(12)],
    'vit_32': [f'hs_{n}' for n in range(12)]
}

feature_types = [
    None, 'taxonomic', 'encyclopaedic', 'function', 'visual_perceptual',
    'other_perceptual'
]

uni_models = {
    'vit_16': {'dnn': 'vit_16', 'stream': 'img', 'hs_type': None}, 
    'vit_32': {'dnn': 'vit_32', 'stream': 'img', 'hs_type': None}, 
    'gpt_concepts': {'dnn': 'gpt', 'stream': 'txt', 'hs_type': 'concepts'},
    'gpt_definition': {'dnn': 'gpt', 'stream': 'txt', 'hs_type': 'definition'}, 
    'bert_concepts': {'dnn': 'bert', 'stream': 'txt', 'hs_type': 'concepts'},
    'bert_definition': {'dnn': 'bert', 'stream': 'txt', 'hs_type': 'definition'}
}