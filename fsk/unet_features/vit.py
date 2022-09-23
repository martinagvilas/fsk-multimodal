import torch
from timm import create_model
from timm.data import create_transform, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTModel

from fsk.dataprep.dataloader import ThingsDataset


class FtVit():
    def __init__(self, version, project_path, result_dir: str = 'net_ft', batch_idx=None, device='cpu'):
        self.version = version
        self.model_name = f'vit_{version}'
        self.layers = {'img': [f'encoder.layer.{n}' for n in range(12)]}
        self.device = device

        self.dataset_path = project_path / 'dataset'
        self.res_path = project_path / 'results' / self.model_name / result_dir
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


class ClsVit(FtVit):
    def __init__(self, version, project_path, result_dir='net_cls', batch_size=32, batch_idx=None, device='cpu'):
        self.batch_size = batch_size
        super().__init__(version, project_path, result_dir, batch_idx, device)

    def get_dataset(self, batch_idx):
        ds = ThingsDataset(self.dataset_path, batch_idx=batch_idx)
        self.dataset = DataLoader(ds, batch_size=self.batch_size,
                                  collate_fn=lambda x: (
                                      torch.stack([self.img_transform(i['img']) for i in x]),
                                      [i['img_id'] for i in x])
                                  )

    def load_model(self):
        if self.version == '32':
            model_name = 'vit_base_patch32_224'
        elif self.version == '16':
            model_name = 'vit_base_patch16_224'
        self.img_transform = create_transform(input_size=224,
                                              interpolation='bicubic',
                                              crop_pct=0.9,
                                              mean=IMAGENET_INCEPTION_MEAN,
                                              std=IMAGENET_INCEPTION_STD,
                                              is_training=False)
        self.model = create_model(model_name, pretrained=True).to(self.device).eval()

    def compute(self):
        logits, img_idx = [], []
        for img, img_id in tqdm(self.dataset, total=len(self.dataset)):
            with torch.no_grad():
                out = self.model(img.to(self.device))
                logits.append(out.cpu())
                img_idx.extend(img_id)
        logits = torch.concat(logits)
        torch.save(logits, self.res_path / 'logits.pt')
        torch.save(img_idx, self.res_path / 'image_ids.pt')
