from collections import OrderedDict

import clip
from tqdm import tqdm
import torch

from fsk.it_match.itm import ItmModel, add_feature_extractor


class ItmClip(ItmModel):
    def __init__(self, model_name, project_path, batch_idx=None, device='cpu'):
        ItmModel.__init__(self, model_name, project_path, batch_idx, device)
        self.layers = {
            'txt': [f'transformer.resblocks.{i}' for i in range(12)],
            'img': [f'visual.transformer.resblocks.{i}' for i in range(12)]
        }
        self.load_model()
    
    def load_model(self):
        model, img_transform = clip.load('ViT-B/32', self.device)
        model.to(self.device)
        model.eval()
        # add feature extractor for getting hidden states
        layers = [l for t in self.layers.values() for l in t]
        model = add_feature_extractor(model, layers) 
        self.model = model
        self.img_transform = img_transform
        return

    def compute(self):
        print("Computing Image-Text matching using CLIP model")
        concepts_ft = self.get_txt_ft(txt_type='concepts')
        features_ft = self.get_txt_ft(txt_type='sem_features')
        for _, data in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            img_id = data['img_id']
            c_match_file = self.res_paths['concept_match'] / f'{img_id}.pt'
            sf_match_file = self.res_paths['feature_match'] / f'{img_id}.pt'
            if c_match_file.is_file() and sf_match_file.is_file():
                continue
            img_ft = self.encode_image(data['img'], img_id)
            concept_match = self.compute_match(img_ft, concepts_ft)
            torch.save(concept_match, c_match_file)
            sft_match = self.compute_match(img_ft, features_ft)
            torch.save(sft_match, sf_match_file)

    def get_txt_ft(self, txt_type='concepts', overwrite=False):
        out_file = self.res_paths['net_ft'] / f'c-out_txt_{txt_type}.pt'
        if out_file.is_file() and overwrite == False:
            txt_ft = torch.load(out_file).to(self.device)
        else:
            print(f"Computing {txt_type} embeddings", flush=True)
            if txt_type == 'concepts':
                txt = self.concepts
            elif txt_type == 'sem_features':
                txt = [
                    ((" ").join(t.split("_"))).capitalize() + '.' 
                    for t in self.sem_features
                ]
            self.model.__features__ = OrderedDict()
            tokens = clip.tokenize(txt).to(self.device)
            # Save EOS token
            tokens_idxs = (tokens == torch.tensor(49407)).nonzero()
            with torch.no_grad():
                txt_ft = self.model.encode_text(tokens)
            torch.save(txt_ft, out_file)
            # Save hidden states
            hs = []
            for l in self.layers['txt']:
                hs.append(self.model.__features__[l])
            hs = torch.permute(torch.stack(hs), (2, 0, 1, 3))
            hs = hs[tokens_idxs[:,0], :, tokens_idxs[:, 1], :]
            hs_file = self.res_paths['net_ft'] / f'hs_txt_{txt_type}.pt'
            torch.save(hs, hs_file)
        return txt_ft

    def encode_image(self, img, img_id):
        img = torch.unsqueeze(self.img_transform(img), dim=0).to(self.device)
        self.model.__features__ = OrderedDict()
        with torch.no_grad():
            img_ft = self.model.encode_image(img)
            out_file = self.res_paths['net_ft'] / f'c-out_img_{img_id}.pt'
            torch.save(img_ft, out_file)
            hs = []
            for l in self.layers['img']:
                hs.append(self.model.__features__[l])
            hs = torch.stack(hs)
            hs_file = self.res_paths['net_ft'] / f'hs_img_{img_id}.pt'
            torch.save(hs, hs_file)
        return img_ft

    def compute_match(self, img_ft, txt_ft):
        img_ft /= img_ft.norm(dim=-1, keepdim=True)
        txt_ft /= txt_ft.norm(dim=-1, keepdim=True)
        match = torch.squeeze((img_ft @ txt_ft.T), dim=0)
        return match


def compute(project_path, device):
    print("Computing Image-Text matching using CLIP model")
    itm = ItmClip('clip', project_path, device=device)
    itm.compute()
    return 
