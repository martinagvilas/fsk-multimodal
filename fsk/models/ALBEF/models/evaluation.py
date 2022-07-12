import pytorch_lightning as pl


class Evaluation(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass
