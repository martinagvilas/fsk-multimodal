# Semantic feature type information
feature_types = [
    None, 'taxonomic', 'encyclopaedic', 'function', 'visual_perceptual',
    'other_perceptual'
]

models = {
    'clip_img': {'dnn': 'clip', 'stream': 'img'},
    'clip_txt': {'dnn': 'clip', 'stream': 'txt'},
    'albef_img': {'dnn': 'albef', 'stream': 'img'},
    'albef_txt': {'dnn': 'albef', 'stream': 'txt'},
    'albef_multi': {'dnn': 'albef', 'stream': 'multi'},
    'vilt_multi': {'dnn': 'vilt', 'stream': 'multi'},
    'vit_16': {'dnn': 'vit_16', 'stream': 'img'}, 
    'vit_32': {'dnn': 'vit_32', 'stream': 'img'}, 
    'gpt': {'dnn': 'gpt', 'stream': 'txt'},
    'bert': {'dnn': 'bert', 'stream': 'txt'},
}


# Layer information
layers = {
    'albef_img': [f'hs_{n}' for n in range(12)] + ['c-out'],
    'albef_txt': [f'hs_{n}' for n in range(6)] + ['c-out'],
    'albef_multi': [f'hs_{n}' for n in range(6)],
    'clip_img': [f'hs_{n}' for n in range(12)] + ['c-out'],
    'clip_txt': [f'hs_{n}' for n in range(12)] + ['c-out'],
    'vilt_multi': [f'hs_{n}' for n in range(12)],
    'bert': [f'hs_{n}' for n in range(12)],
    'gpt': [f'hs_{n}' for n in range(12)],
    'vit_16': [f'hs_{n}' for n in range(12)],
    'vit_32': [f'hs_{n}' for n in range(12)]
}

# Models information
multi_models_info = [
    {'dnn': 'clip', 'stream': ['img', 'txt']},
    {'dnn': 'albef', 'stream': ['img', 'txt', 'multi']},
    {'dnn': 'vilt', 'stream': ['multi']}
]

multi_models = []
for m_info in multi_models_info:
    model = m_info['dnn']
    for stream in m_info['stream']:
        for layer in layers[f'{model}_{stream}']:
            multi_models.append([model, stream, layer])

uni_models = {
    'vit_16': {'dnn': 'vit_16', 'stream': 'img', 'hs_type': None}, 
    'vit_32': {'dnn': 'vit_32', 'stream': 'img', 'hs_type': None}, 
    'gpt': {'dnn': 'gpt', 'stream': 'txt', 'hs_type': None},
    'bert': {'dnn': 'bert', 'stream': 'txt', 'hs_type': None},
}
