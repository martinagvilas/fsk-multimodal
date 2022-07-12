import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor, ModelCheckpoint, EarlyStopping, \
    ModelSummary
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.utilities.seed import seed_everything

from cka_clip.utils.logging import init_wandb
from cka_clip.utils.misc import init_logger, Config
from dataset import COCO17DataModule
from .dataset import train_transform, test_transform, clean_caption
from .models.module import ALBEFModule


class Trainer:
    def __init__(self, config: Config, debug: bool = False):
        self.config = config
        self.logger = init_logger(type(self).__name__, stream=True, file=False)

        self.debug = debug

        self.nr_gpus = len(self.config.general.gpus) if isinstance(self.config.general.gpus, list) else -1

        self.module = None
        self.datamodule = None
        self.trainer = None

    def prepare_workspace(self):
        self.logger.info("Prepare workspace")

        checkpoint_dir = Path(self.config.general._checkpoint_path)
        checkpoint_dir.mkdir(exist_ok=True)
        if not self.nr_gpus == -1:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.config.general.gpus))
        torch.cuda.empty_cache()
        self.logger.info("Number of GPUS: {0}".format(torch.cuda.device_count()))

        seed_everything(self.config.general.seed)

    def configure_callbacks(self):
        self.logger.info("Configure callbacks")
        callback = [
            ModelSummary(max_depth=3 if self.debug else 1),
            ModelCheckpoint(save_last=True,
                            save_top_k=1,
                            monitor="val_loss",
                            mode='min',
                            verbose=True,
                            save_weights_only=False,
                            filename=self.config.general.experiment_name + "-{epoch}-{val_loss:.2f}",
                            dirpath=self.config.general._checkpoint_path)
        ]
        if not self.debug:
            callback.append(LearningRateMonitor(logging_interval="step"))
        if self.config.general.show_progress_bar:
            callback.append(RichProgressBar())

        if 'callbacks' in self.config.keys():
            if 'early_stopping' in self.config.callbacks.keys():
                callback.append(EarlyStopping(**self.config.callbacks.early_stopping))

        return callback

    def configure_logger(self):
        self.logger.info('Configure logger')
        if self.debug:
            logger = DummyLogger()
        else:
            logger = init_wandb(config=self.config.flatten(),
                                name=self.config.general.experiment_name,
                                **self.config.logger)
            # logger.watch(self.module, log='all', log_freq=50)
        return logger

    def configure_datamodule(self):

        self.logger.info(f"Configure data")

        def collate_fn(batch):
            images = torch.stack([i[0] for i in batch])
            captions = [clean_caption(i[1], max_words=self.config.data.max_words) for i in batch]
            token_captions = self.module.model.tokenizer(captions, padding='longest',
                                                         max_length=self.config.data.max_words, return_tensors="pt")
            ids = torch.tensor([i[2] for i in batch])
            return images, token_captions, ids

        self.datamodule = COCO17DataModule(expand_labels=False,
                                           target_transform=lambda x: x[0],
                                           train_transform=train_transform(self.config.model.image_res),
                                           test_transform=test_transform(self.config.model.image_res),
                                           collate_fn=collate_fn,
                                           batch_size=self.config.data.batch_size,
                                           num_workers=self.config.data.num_workers,
                                           root=self.config.data.root)

    def configure_module(self):
        self.logger.info(f"Configure module")
        self.module = ALBEFModule(alpha=self.config.trainer.alpha,
                                  warm_up=self.config.trainer.warm_up,
                                  model_kwargs=self.config.model,
                                  optimizer_kwargs=self.config.optimizer,
                                  lr_scheduler_kwargs=self.config.lr_scheduler)

    def configure_trainer(self):
        self.logger.info("Configure trainer")
        callbacks = self.configure_callbacks()

        self.trainer = pl.Trainer(
            num_sanity_val_steps=2,
            max_epochs=self.config.trainer.epochs,
            callbacks=callbacks,
            logger=self.configure_logger(),
            val_check_interval=self.config.trainer.val_check_interval,
            log_every_n_steps=50,
            sync_batchnorm=True,
            precision=self.config.trainer.precision,
            devices='auto' if self.nr_gpus == -1 else self.nr_gpus,
            accelerator='gpu',
            auto_select_gpus=self.nr_gpus == -1,
            num_nodes=1,
            strategy='ddp_sharded',
            detect_anomaly=False,
            fast_dev_run=1 if self.debug else False,
            progress_bar_refresh_rate=1 if self.config.general.show_progress_bar else 0,
            enable_progress_bar=self.config.general.show_progress_bar,
        )

    def prepare(self):
        self.prepare_workspace()
        self.configure_module()
        self.configure_datamodule()
        self.configure_trainer()

    def fit(self):
        self.prepare()
        self.trainer.fit(datamodule=self.datamodule, model=self.module)

        wandb.finish()
        torch.cuda.empty_cache()


def cmd_line_interface():
    parser = argparse.ArgumentParser(description='Start training.')
    parser.add_argument('config_file', help="Path to configuration file. Or path to dir of multiple config files.",
                        type=str)
    parser.add_argument('--debug', help="Do a learning rate search.", action='store_true')
    args = parser.parse_args()

    config_file = Path(args.config_file)
    if config_file.is_file():
        config = Config.from_yaml(config_file)
        trainer = Trainer(config=config, debug=args.debug)
        trainer.fit()
    else:
        raise ValueError(f"Cannot find config path: {config_file}.")
