from nltk.corpus import wordnet as wn
import torch
from transformers import BertTokenizer, BertConfig, BertModel

from fsk.dataprep.utils import get_concepts, get_fsk_synsets


class FtBert():
    def __init__(self, project_path, batch_idx=None, device='cpu'):
        self.model_name = f'bert'
        self.layers = {'txt': [f'encoder.layer.{n}' for n in range(12)]}
        self.device = device

        self.dataset_path = project_path / 'dataset'
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
    
    def get_dataset(self):
        synsets = get_fsk_synsets(self.dataset_path)
        concepts = get_concepts(self.dataset_path)
        # Create dictionary with concept name and synset definition
        self.dataset = {}
        for c, s in zip(concepts, synsets):
            s_wn = wn.synset(('.').join(s.split('-')))
            s_def = f"{c} is a {s_wn.definition()}"
            self.dataset[s] = [c, s_def]
        
    def compute(self):
        for c_idx, caption_type in enumerate(['concept', 'definition']):
            hs_cls = []
            hs_word = []
            for s_vals in self.dataset.values():
                tokens = self.processor(
                    s_vals[c_idx], padding=True, return_tensors="pt"
                )
                with torch.no_grad():
                    out = self.model(**tokens)
                hs = torch.squeeze(torch.stack(out.hidden_states))[1:]
                hs_cls.append(torch.squeeze(hs[:, 0, :]))
                hs_word.append(torch.squeeze(hs[:, 1, :]))
            hs_cls_file = self.res_path / f'hs_txt_{caption_type}.pt'
            torch.save(torch.stack(hs_cls), hs_cls_file)
            hs_word_file = self.res_path / f'hs_txt_{caption_type}.pt'
            torch.save(torch.stack(hs_word), hs_word_file)