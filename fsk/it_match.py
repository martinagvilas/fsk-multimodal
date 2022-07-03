import argparse
from collections import OrderedDict
from pathlib import Path

import clip
import torch
from torch.utils.data import DataLoader

from fsk.dataprep.dataloader import ThingsDataset
from fsk.dataprep.utils import get_fsk_concepts, get_fsk_features


class ItmModel:
    def __init__(
        self, model_name, project_path, batch_idx=None, device='cpu'
    ):
        self.model_name = model_name
        self.device = device
        self.project_path = project_path
        self._get_paths()
        self.concepts = get_fsk_concepts(self.dataset_path)
        self.sem_features = self._get_sem_features()
        self.dataset = self._get_dataset(batch_idx)
    
    def _get_paths(self):
        self.dataset_path = self.project_path / 'dataset'
        self.things_path = self.dataset_path / 'things'
        self.annot_path = self.dataset_path / 'annotations'
        res_path = self.project_path / f'results/{self.model_name}'
        results_folders = ['concept_match', 'feature_match', 'net_ft']
        results_paths = {}
        for rf in results_folders:
            rf_path = res_path / rf
            rf_path.mkdir(parents=True, exist_ok=True)
            results_paths[rf] = rf_path
        self.results_paths = results_paths
        return

    def _get_sem_features(self):
        sem_features = get_fsk_features(self.dataset_path)
        return sem_features

    def _get_dataset(self, batch_idx):
        ds = ThingsDataset(self.dataset_path, batch_idx=batch_idx)
        dataloader = DataLoader(
            ds, batch_size=None, batch_sampler=None, collate_fn=lambda x: x
        )
        return enumerate(dataloader)


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

    def compute_match(self):
        concepts_ft = self._get_txt_ft(txt_type='concepts')
        features_ft = self._get_txt_ft(txt_type='sem_features')
        for _, data in self.dataset:
            img_id = data['img_id']
            c_match_file = self.results_paths['concept_match'] / f'{img_id}.pt'
            sf_match_file = self.results_paths['feature_match'] / f'{img_id}.pt'
            if c_match_file.is_file() and sf_match_file.is_file():
                continue
            img_ft = self._encode_image(data['img'], img_id)
            concept_match = self._compute_match(img_ft, concepts_ft)
            torch.save(concept_match, c_match_file)
            sft_match = self._compute_match(img_ft, features_ft)
            torch.save(sft_match, sf_match_file)

    def _get_txt_ft(self, txt_type='concepts', overwrite=False):
        out_file = self.results_paths['net_ft'] / f'out_txt_{txt_type}.pt'
        if out_file.is_file() and overwrite == False:
            txt_ft = torch.load(out_file)
        else:
            print(f"Computing {txt_type} embeddings", flush=True)
            if txt_type == 'concepts':
                txt = self.concepts
            elif txt_type == 'sem_features':
                txt = self.sem_features
            self.model.__features__ = OrderedDict()
            tokens = clip.tokenize(txt)
            with torch.no_grad():
                txt_ft = self.model.encode_text(tokens)
            torch.save(txt_ft, out_file)
            # Save hidden states
            hs = []
            for l in self.layers['txt']:
                hs.append(self.model.__features__[l])
            hs = torch.stack(hs)
            hs_file = self.results_paths['net_ft'] / f'hs_txt_{txt_type}.pt'
            torch.save(hs, hs_file)
        return txt_ft

    def _encode_image(self, img, img_id):
        img = torch.unsqueeze(self.img_transform(img), dim=0)
        self.model.__features__ = OrderedDict()
        with torch.no_grad():
            img_ft = self.model.encode_image(img)
            out_file = self.results_paths['net_ft'] / f'out_img_{img_id}.pt'
            torch.save(img_ft, out_file)
            hs = []
            for l in self.layers['img']:
                hs.append(self.model.__features__[l])
            hs = torch.stack(hs)
            hs_file = self.results_paths['net_ft'] / f'hs_img_{img_id}.pt'
            torch.save(hs, hs_file)
        return img_ft

    def _compute_match(self, img_ft, txt_ft):
        img_ft /= img_ft.norm(dim=-1, keepdim=True)
        txt_ft /= txt_ft.norm(dim=-1, keepdim=True)
        match = torch.squeeze((img_ft @ txt_ft.T), dim=0)
        return match


def add_feature_extractor(model, layers):
    def get_activation(layer_name):
        def hook(_, input, output):
            model.__features__[layer_name] = output.detach()
        return hook
    for layer_name, layer in model.named_modules():
        if layer_name in layers:
            layer.register_forward_hook(get_activation(layer_name))
    return model


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', action='store', required=True)
    parser.add_argument('-pp', action='store', required=True)
    parser.add_argument('-id', action='store', required=False)
    model_name = parser.parse_args().m
    project_path = Path(parser.parse_args().pp)
    batch_idx = parser.parse_args().id

    itm = ItmClip(model_name, project_path, batch_idx)
    itm.compute_match()

