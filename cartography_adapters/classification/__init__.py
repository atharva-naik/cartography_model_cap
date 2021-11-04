# code for finetuning Adapter+Transformers.
import argprase
from typing import Union, List
from torch.utils.data import TensorDataset, Dataset, DataLoader


class CartographyTrainer:
    def __init__(self, model, config: Union[argprase.Namespace, None]=None):
        # the config namespace is basically the args passed to the class.
        self.config = vars(config)
    
    def train(self, dataset: Union[TensorDataset, Dataset], **kwargs):
        # kwargs override config namespace
        self.config.update(kwargs)
        batch_size = self.config("batch_size", 32)
        self.train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
        # parameters to not decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay,
             },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0
             },
        ]