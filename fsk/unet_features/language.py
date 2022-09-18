import pickle

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
import torch
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import GPT2Tokenizer, GPT2Model

from fsk.dataprep.utils import get_concepts, get_synsets_ids


class FtLanguage():
    def __init__(self, project_path, text_type):
        self.dataset_path = project_path / 'dataset'
        self.text_type = text_type

    def get_dataset(self):
        if self.text_type == 'concept':
            self.dataset = self.get_concepts_dataset()
        elif self.text_type == 'captions':
            file = self.dataset_path / 'annotations' / 'cc_captions.pkl'
            self.dataset = pickle.load(file)
        
    def get_concepts_dataset(self):
        synsets_ids, concepts = get_synsets_ids(self.dataset_path)
        synsets = list(synsets_ids.keys())
        concepts = concepts['things']
        dataset = {}
        for c, s in zip(concepts, synsets):
            s_name = ('.').join(s.split('-'))
            s_wn = wn.synset(s_name)
            s_def = s_wn.definition()
            definition = [f"{c} is a {s_def}"]
            c_art = 'an' if c.startswith(('a','e','i','o','u')) else 'a'
            q = [
                f'what is {c_art} {c}?', 
                f'how can I describe {c_art} {c}?',
                f'what do you refer by {c_art} {c}?'
            ]
            others = [
                f'a very standard {c}', f'a photo of one {c}', f'this is my {c}',
                f'i can see {c_art} {c}', f'a close up of {c_art} {c}',
                f'this {c} is very typical'
            ]
            dataset[s] = [c] + definition + q + others
        self.synsets = synsets
        return dataset
    
    # def get_cc_captions(self):
    #     return captions


class FtGPT(FtLanguage):
    def __init__(self, project_path, text_type, device='cpu'):
        FtLanguage.__init__(self, project_path, text_type)
        self.model_name = 'gpt'
        self.layers = {'txt': [f'h.{n}' for n in range(12)]}        
        self.device = device

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
    
    def compute(self):
        hs = []
        for s in self.synsets:
            txt = [t + " <|endoftext|>" for t in self.dataset[s]]
            s_hs = []
            for t in txt:
                tokens = self.processor(
                    t, truncation=True, return_tensors="pt"
                )
                with torch.no_grad():
                    out = self.model(**tokens)
                # Use EOS token as CLIP
                s_hs.append(
                    torch.squeeze(torch.stack(out.hidden_states)[1:, :, -1, :])
                )
            hs.append(torch.mean(torch.stack(s_hs), dim=0))
        hs_file = self.res_path / f'hs_txt_concepts.pt'
        torch.save(torch.stack(hs), hs_file)


class FtBert(FtLanguage):
    def __init__(self, project_path, text_type, device='cpu'):
        FtLanguage.__init__(self, project_path, text_type)
        self.model_name = f'bert'
        self.layers = {'txt': [f'encoder.layer.{n}' for n in range(12)]}
        self.device = device

        self.res_path = project_path / 'results' / self.model_name / 'net_ft'
        self.res_path.mkdir(parents=True, exist_ok=True)
    
        self.load_model()
        self.get_dataset()
    
    def load_model(self):
        self.processor = BertTokenizer.from_pretrained('bert-base-uncased')
        cfg = BertConfig.from_pretrained(
            'bert-base-uncased', output_hidden_states=True
        )
        self.model = BertModel.from_pretrained('bert-base-uncased', config=cfg)
    
    def compute(self):
        hs = []
        for s in self.synsets:
            tokens = self.processor(
                self.dataset[s], padding=True, return_tensors="pt"
            )
            with torch.no_grad():
                out = self.model(**tokens)
            # Use CLS token as ALBEF and VILT
            s_hs = torch.stack(out.hidden_states)[1:, :, 0, :]
            s_hs = torch.mean(s_hs, dim=1)
            hs.append(s_hs)
        hs_file = self.res_path / f'hs_txt_concepts.pt'
        torch.save(torch.stack(hs), hs_file)