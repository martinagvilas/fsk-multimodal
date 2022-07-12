import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .model_retrieval import ALBEF
from ..optim import create_optimizer
from ..scheduler import WarmupCosineAnnealingLR


class ALBEFModule(pl.LightningModule):
    def __init__(self,
                 alpha,
                 warm_up,
                 model_kwargs,
                 optimizer_kwargs,
                 lr_scheduler_kwargs,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.model = type(self).__name__

        self.model = ALBEF.from_cktp(model_kwargs)

    def training_step(self, train_batch, batch_idx):
        image, text, idx = train_batch

        if self.current_epoch > 0 or not self.hparams.warm_up:
            alpha = self.hparams.alpha
        else:
            alpha = self.hparams.alpha * min(1, batch_idx / self.trainer.num_training_batches)

        loss_ita, loss_itm = self.model(image, text, alpha=alpha, idx=idx)
        loss = loss_ita + loss_itm

        self.log_dict(dict(loos=loss_itm+loss_ita, loss_itm=loss_itm, loss_ita=loss_ita), sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        image, text, idx = val_batch
        loss_ita, loss_itm = self.model(image, text, alpha=self.hparams.alpha, idx=idx)

        bs = image.shape[0]
        result = self.model.similarity_and_matching(image, text, pairwise=True)['score']

        ground_truth = torch.arange(bs).to(self.device)
        acc_i = (torch.argmax(result, 1) == ground_truth).sum() / bs
        acc_t = (torch.argmax(result, 0) == ground_truth).sum() / bs

        self.log_dict(
            {'val_loss': loss_itm + loss_ita, 'loss_itm': loss_itm, 'loss_ita': loss_ita, 'val_top1_acc_image': acc_i,
             'val_top1_acc_text': acc_t},
            prog_bar=True,
            sync_dist=True)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams.optimizer_kwargs, self.model)

        warmup_steps = int(self.hparams.lr_scheduler_kwargs.pop('warmup_epochs') * (
                self.estimated_stepping_batches // self.trainer.max_epochs))
        scheduler = {
            "scheduler": WarmupCosineAnnealingLR(optimizer,
                                                 warmup_steps=warmup_steps,
                                                 max_steps=self.estimated_stepping_batches,
                                                 **self.hparams.lr_scheduler_kwargs),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    @property
    def estimated_stepping_batches(self):
        effective_accum = self.trainer.accumulate_grad_batches * self.trainer.num_devices
        batches = len(self.trainer.datamodule.train_dataloader())
        return (batches // effective_accum) * self.trainer.max_epochs
