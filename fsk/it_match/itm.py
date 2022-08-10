from torch.utils.data import DataLoader

from fsk.dataprep.dataloader import ThingsDataset
from fsk.dataprep.utils import get_concepts, get_fsk_features


class ItmModel:
    def __init__(
        self, model_name, project_path, batch_idx=None, device='cpu'
    ):
        self.model_name = model_name
        self.device = device
        self.project_path = project_path
        self._get_paths()
        self.concepts = get_concepts(self.dataset_path)
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
        self.res_paths = results_paths
        return

    def _get_sem_features(self):
        sem_features = get_fsk_features(self.dataset_path)
        return sem_features

    def _get_dataset(self, batch_idx):
        ds = ThingsDataset(self.dataset_path, batch_idx=batch_idx)
        dataloader = DataLoader(
            ds, batch_size=None, batch_sampler=None, collate_fn=lambda x: x
        )
        return dataloader


def add_feature_extractor(model, layers):
    def get_activation(layer_name):
        def hook(_, input, output):
            model.__features__[layer_name] = output.detach()
        return hook
    for layer_name, layer in model.named_modules():
        if layer_name in layers:
            layer.register_forward_hook(get_activation(layer_name))
    return model

