# Layer information
layers = {
    'albef_img': [f'hs_{n}' for n in range(12)] + ['c_out'],
    'albef_multi': [f'hs_{n}' for n in range(6)],
    'clip_img': [f'hs_{n}' for n in range(12)] + ['c_out'],
    'vilt_multi': [f'hs_{n}' for n in range(12)]
}