from fsk.dataprep.utils import get_synsets_ids
from fsk.human_similarity.distances import get_mcrae_distances


class RSA():
    def __init__(
        self, project_path, model_1, model_2, dist_l1='cosine', 
        dist_l2='spearman'
    ):
        self.project_path = project_path
        self.dataset_path = project_path / 'dataset'
        # Get relevant information
        self.models_info = self._get_models_info(model_1, model_2)
        # Get concepts information
        self.synsets_ids, self.concepts = get_synsets_ids(self.dataset_path)
        self.synsets = list(self.synsets_ids.keys())
        # Determine distance metric at level 1 and level 2
        self.dist_metric_l1 = dist_l1
        self.dist_metric_l2 = dist_l2
        # Load or compute distances
        self.dist_model_1 = self.get_distances()

    def _get_models_info(self, model_1, model_2):
        models_info = {0: {}, 1:{}}
        for idx, model in enumerate([model_1, model_2]):
            if model == 'sem':
                models_info[idx]['name'] = 'semantic_mcrae'
            else:
                models_info[idx]['name'] = model
                model_parts = model.split('_')
                models_info[idx]['dnn'] = model_parts[0]
                models_info[idx]['stream'] = model_parts[1]
                # add layers and layer index
        return models_info

    def get_distances(self):
        for model in self.models_info.values():
            if model['name'] == 'semantic_mcrae':
                dist = get_mcrae_distances(
                    self.project_path, self.synsets, self.concepts['mcrae']
                )
            pass
        #if self.model
        #for synset in self.synsets:
            
        # TODO: compute distances for every layer (as dictionary)
        ## probably best to compute and add?

        print('done')