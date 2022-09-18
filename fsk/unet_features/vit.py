from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTModel

from fsk.dataprep.dataloader import ThingsDataset


class FtVit():
    def __init__(self, version, project_path, batch_idx=None, device='cpu'):
        self.version = version
        self.model_name = f'vit_{version}'
        self.layers = {'img': [f'encoder.layer.{n}' for n in range(12)]}
        self.device = device

        self.dataset_path = project_path / 'dataset'
        self.res_path = project_path / 'results' / self.model_name / 'net_ft'
        self.res_path.mkdir(parents=True, exist_ok=True)
    
        self.load_model()
        self.get_dataset(batch_idx)
    
    def load_model(self):
        if self.version == '32':
            source = 'google/vit-base-patch32-224-in21k'
        elif self.version == '16':
            source = 'google/vit-base-patch16-224-in21k'
        self.img_transform = ViTFeatureExtractor.from_pretrained(source)
        self.model = ViTModel.from_pretrained(
            source, output_hidden_states=True
        ).to(self.device).eval()

    def get_dataset(self, batch_idx):
        ds = ThingsDataset(self.dataset_path, batch_idx=batch_idx)
        self.dataset = DataLoader(
            ds, batch_size=None, batch_sampler=None, collate_fn=lambda x: x
        )

    def compute(self):
        for _, data in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            img_ft = self.img_transform(
                data['img'], return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**img_ft)
                hs = torch.squeeze(torch.stack(out.hidden_states)[1:, :, 0, :])
                hs_file = self.res_path / f"hs_img_{data['img_id']}.pt"
                torch.save(hs, hs_file)

