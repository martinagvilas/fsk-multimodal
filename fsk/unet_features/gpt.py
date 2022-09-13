from nltk.corpus import wordnet as wn
import torch
from transformers import GPT2Tokenizer, GPT2Model

from fsk.dataprep.utils import get_concepts, get_fsk_synsets, get_synsets_ids


class FtGPT():
    def __init__(self, project_path, caption_type, device='cpu'):
        self.model_name = 'gpt'
        self.caption_type = caption_type
        self.layers = {'txt': [f'h.{n}' for n in range(12)]}        
        self.device = device

        self.dataset_path = project_path / 'dataset'
        self.res_path = project_path / 'results' / self.model_name / 'net_ft'
        self.res_path.mkdir(parents=True, exist_ok=True)
    
        self.load_model()
        self.get_dataset()

    def load_model(self):
        model_path = self.dataset_path / 'models_config/gpt2/'
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.processor = tokenizer
        self.model = GPT2Model.from_pretrained(
            model_path, output_hidden_states=True
        )
    
    def get_dataset(self):
        synsets = get_fsk_synsets(self.dataset_path)
        concepts = get_concepts(self.dataset_path)
        # Create dictionary with concept name and synset definition
        self.dataset = {}
        for c, s in zip(concepts, synsets):
            s_wn = wn.synset(('.').join(s.split('-')))
            if self.caption_type == 'definition':
                data = f"{c} is a {s_wn.definition()}"
            else:
                data = c
            self.dataset[s] = data
    
    def compute(self):
        hs = []
        for s_vals in self.dataset.values():
            txt = s_vals[c_idx] + "<|endoftext|>"
            tokens = self.processor(
                txt, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                out = self.model(**tokens)
            # Use EOS token as CLIP
            hs.append(
                torch.squeeze(torch.stack(out.hidden_states)[1:, :, -1, :])
            )
        hs_file = self.res_path / f'hs_txt_{self.caption_type}.pt'
        torch.save(torch.stack(hs), hs_file)