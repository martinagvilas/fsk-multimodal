# Layer information
layers = {
    'albef_img': [f'hs_{n}' for n in range(12)] + ['c-out'],
    'albef_txt': [f'hs_{n}' for n in range(6)] + ['c-out'],
    'albef_multi': [f'hs_{n}' for n in range(6)],
    'clip_img': [f'hs_{n}' for n in range(12)] + ['c-out'],
    'clip_txt': [f'hs_{n}' for n in range(12)] + ['c-out'],
    'vilt_multi': [f'hs_{n}' for n in range(12)],
    'bert': [f'hs_{n}' for n in range(12)],
    'gpt': [f'hs_{n}' for n in range(12)]
}