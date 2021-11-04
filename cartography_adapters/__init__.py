# code for tracking training dynamics for transformers+adapters.
import os
# comment this out except for KGP servers.
os.environ['OPENBLAS_NUM_THREADS'] = "12"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (BertModel,
                          BertConfig,
                          RobertaModel,
                          RobertaConfig,
                          BertForSequenceClassification,
                          RobertaForSequenceClassification)


class RobertaForCartography(RobertaForSequenceClassification):
    '''analogus to AdaptedRobertaForSequenceClassification. Includes support for adapters.'''
    def __init__(self, config=None):
        # if no config is supplied use the default roberta config (which is modified by adapter transformers.)
        if config is None: config = RobertaConfig()
        super(RobertaForCartography, self).__init__(config)
        
    def forward(self, *args, **kwargs):
        outputs = self.roberta(*args, **kwargs)
        print(type(outputs))
        exit("BYE")
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs  # Modified from original `Transformers` since we need sequence output to summarize.
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
            
        return outputs  # (loss), logits, sequence_output, pooled_sequence_output, (hidden_states), (attentions)

    
def main(**args):
    from torch.utils.data import DataLoader
    from transformers import RobertaTokenizer
    try:
        from cartography_adapters.datautils import GLUEDataset
    except ImportError:
        from .datautils import GLUEDataset
    # number of worker threads to be used for dataloading.
    num_worker = args.get("num_workers", 1)
    # batch size for training
    batch_size = args.get("batch_size", 32)
    # config = RobertaConfig()
    model = RobertaForCartography()
    # print(model)
    tokenizer = RobertaTokenizer.from_pretrained("../roberta-base-tok")
    trainset = GLUEDataset(
        path="./data/MNLI/original/multinli_1.0_train.jsonl",
        tokenizer=tokenizer, 
        task_name="MNLI",
    )
    trainloader = DataLoader(trainset, num_workers=num_workers, 
                             shuffle=True, batch_size=batch_size)
    # iterate over trainloader.
    for step, batch in enumerate(trainloader):
        print("id:", batch[0])
        print("label:", batch[-1])
        print("input_ids:", batch[1])
        print("attention_mask:", batch[2])
        model(input_ids=batch[1], attention_mask=batch[2])
        break
        
    
if __name__ == "__main__":
    main()