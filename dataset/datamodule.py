import torch
import pytorch_lightning as pl

class Datamodule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        root = "./"
        inbreast_path = "./"
        mias_path = "./"
        ddsn_path = "./"
        