import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, \
    InterpolationMode

from fsk.it_match.itm import ItmModel, add_feature_extractor
from fsk.models.ALBEF.models.model_retrieval import ALBEF


class ItmAlbef(ItmModel):
    def __init__(self, model_name, project_path, batch_idx=None, device='cpu'):
        ItmModel.__init__(self, model_name, project_path, batch_idx, device)
        self.layers = {
            'txt': [f'text_encoder.encoder.layer.{n}' for n in range(12)],
            'img': [f'visual_encoder.blocks.{n}' for n in range(12)]
        }
        ## TODO: define these and assert that the length of hs is the same
        self.load_model()
    
    def load_model(self):
        config = dict(
            checkpoint_path=(self.dataset_path/'models_config/albef_mscoco.pth'),
            bert_config=(self.dataset_path / 'models_config/config_bert.json'),
            bert_dir=(self.dataset_path/'models_config/bert/'),
            cache_dir='~/.cache/huggingface/transformers', 
            image_res=384,
            queue_size=65536,
            momentum=0.995,
            vision_width=768,
            embed_dim=256,
            temp=0.07,
            distill=False,
            checkpoint_vit=False,
            device=self.device,
        )
        model = ALBEF.from_cktp(config)
        model.to(config['device'])
        model.eval()
        img_transform = Compose([
            Resize(
                (config['image_res'], config['image_res']), 
                interpolation=InterpolationMode.BICUBIC
            ),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            ),
        ])
        model = add_feature_extractor(model, self.layers['img'])
        self.model = model
        self.img_transform = img_transform
        return model, img_transform
    
    def compute_match(self):
        for _, data in self.dataset:
            # Check file doesn't exist
            img_id = data['img_id']
            c_match_file = self.results_paths['concept_match'] / f'{img_id}.pt'
            sf_match_file = self.results_paths['feature_match'] / f'{img_id}.pt'
            if c_match_file.is_file() and sf_match_file.is_file():
                continue
            # Compute image features
            img_ft = torch.unsqueeze(
                self.img_transform(data['img']).to(self.device), dim=0
            )
            # Compute concept match
            concept_match = self._compute_match(img_ft, data, 'concepts')
            torch.save(concept_match, c_match_file)
            # Compute feature match
            sft_match = self._compute_match(img_ft, data, 'sem_features')
            torch.save(sft_match, sf_match_file)


    def _compute_match(self, img_ft, data, txt_type='concepts'):
        # Get text from concepts or features            
        if txt_type == 'concepts':
            txt = self.concepts
            info_key = 'concepts_things'
        elif txt_type == 'sem_features':
            txt = self.sem_features
            info_key = 'features'
        # Determine if text hidden states and contrastive projections files
        # exist or should be computed
        hs_txt_file = self.results_paths['net_ft'] / f'hs_{txt_type}.pt'
        cout_txt_file = self.results_paths['net_ft'] / f'c_out_{txt_type}.pt'
        if hs_txt_file.is_file():
            compute_txt_hs = False
        else:
            compute_txt_hs = True
            txt_hs = []
            txt_c_out = []
        # Define image hidden states and projections file
        img_id = data['img_id']
        hs_img_file = self.results_paths['net_ft'] / f'hs_img_{img_id}.pt'
        cout_img_file = self.results_paths['net_ft'] / f'c_out_img_{img_id}.pt'
        # Compute match and hidden states
        match = []
        for t in txt:
            # Compute text features
            t_prepro = (" ").join(t.split("_"))
            t_ft = self.model.tokenizer(
                t_prepro, adding='max_length', truncation=True, max_length=60, 
                padding=True, return_tensors='pt'
            )
            # Compute match
            self.model.__features__ = OrderedDict()
            with torch.no_grad():
                out = self.model.similarity_and_matching(
                    img_ft.to(device), t_ft.to(device), pairwise=True
                )
            match.append(out['score'])
            # Save visual stream hidden states and contrastive projections 
            # if file doesn't exist
            if ~ hs_img_file.is_file():
                img_hs = []
                for l in self.layers['img']:
                    img_hs.append(self.model.__features__[l])
                torch.save(torch.stack(img_hs), hs_img_file)
                torch.save(torch.squeeze(out['c_out']['img']), cout_img_file)
            # Get text stream hidden states and contrastive projections 
            # if file doesn't exist
            if compute_txt_hs == True:
                txt_hs.append(torch.squeeze(out['hs']['txt'][:, :, 0, :]))
                txt_c_out.append(torch.squeeze(out['c_out'][c]))
            # Get hidden states
            if t in data[info_key]:
                # Get visual stream hidden states
                t_img_hs = []
                for l in self.layers['img']:
                    t_img_hs.append(self.model.__features__[l])
                hs['img'].append(torch.squeeze(torch.stack(t_img_hs)))
                # Get text stream hidden states
                hs['txt'].append(torch.squeeze(out['hs']['txt'][:, :, 0, :]))
                # Get multimodal stream hidden states
                hs['multi'].append(torch.squeeze(out['hs']['multi'][:, :, 0, :]))
                # Get contrastive projections
                for c in c_out.keys():
                    c_out[c].append(torch.squeeze(out['c_out'][c]))
        # Save hidden states
        for key, values in hs.items():
            file = self.results_paths['net_ft'] / f'hs_{key}_{data["img_id"]}.pt'
            torch.save(torch.stack(values), file)
        # Save contrastive projections
        for key, values in c_out.items():
            file = self.results_paths['net_ft'] / f'c_out_{key}_{data["img_id"]}.pt'
            torch.save(torch.stack(values), file)
        # Return match values
        match = torch.squeeze(torch.stack(match))
        return match

        # hs = {hs_type: [] for hs_type in ['img', 'txt', 'multi']}
        # c_out = {out_type: [] for out_type in ['img', 'txt']}

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-pp', action='store', required=True)
    project_path = Path(parser.parse_args().pp)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    itm = ItmAlbef('albef', project_path, device=device)
    itm.compute_match()
