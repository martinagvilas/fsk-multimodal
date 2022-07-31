import argparse
from pathlib import Path

import torch
from transformers import ViltProcessor, ViltForImageAndTextRetrieval

from fsk.it_match.itm import ItmModel


class ItmVilt(ItmModel):
    def __init__(self, model_name, project_path, batch_idx=None, device='cpu'):
        ItmModel.__init__(self, model_name, project_path, batch_idx, device)
        self.layers = {'multi': [f'vilt.encoder.layer.{n}' for n in range(12)]}
        self.load_model()
    
    def load_model(self):
        processor = ViltProcessor.from_pretrained(
            'dandelin/vilt-b32-finetuned-coco'
        )
        model = ViltForImageAndTextRetrieval.from_pretrained(
            'dandelin/vilt-b32-finetuned-coco', output_hidden_states=True
        ).to(self.device)
        model.eval()
        self.model = model
        self.processor = processor
        return model, processor
    
    def compute_match(self):
        for _, data in self.dataset:
            # Check file doesn't exist
            img_id = data['img_id']
            c_match_file = self.res_paths['concept_match'] / f'{img_id}.pt'
            sf_match_file = self.res_paths['feature_match'] / f'{img_id}.pt'
            if c_match_file.is_file() and sf_match_file.is_file():
                continue
            # Compute image features
            img_ft = self.processor.feature_extractor(
                data['img'], return_tensors="pt"
            ).to(self.device)
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
        # Define multimodal hidden states file
        multi_hs_file = (
            self.res_paths['net_ft'] / f'hs_multi_{txt_type}_{data["img_id"]}.pt'
        )
        multi_hs = []
        # Compute match and hidden states
        match = []
        for t in txt:
            # Compute text features
            t_prepro = (" ").join(t.split("_"))
            t_ft = self.processor.tokenizer(
                t_prepro, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            # Compute match
            with torch.no_grad():
                out = self.model(
                    pixel_values=img_ft['pixel_values'], 
                    pixel_mask=img_ft['pixel_mask'], **t_ft
                )
            match.append(out.logits)
            # Get multimodal stream hidden states
            if t in data[info_key]:
                hs = torch.stack(out.hidden_states)[1:, :, 0, :]
                multi_hs.append(torch.squeeze(hs))
        # Save multimodal hidden states
        torch.save(torch.stack(multi_hs), multi_hs_file)
        # Return match values
        match = torch.squeeze(torch.stack(match))
        return match


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-pp', action='store', required=True)
    project_path = Path(parser.parse_args().pp)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    itm = ItmVilt('vilt', project_path, device=device)
    itm.compute_match()
