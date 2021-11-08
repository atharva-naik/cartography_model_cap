import torch
import torch.nn as nn
from transformers import (BertConfig,
                          RobertaConfig,
                          BertModelWithHeads,
                          RobertaModelWithHeads,
                          BertForSequenceClassification, 
                          RobertaForSequenceClassification)  
# class RobertaForCartography(RobertaForSequenceClassification):
#     '''analogus to AdaptedRobertaForSequenceClassification. Includes support for adapters.'''
#     def __init__(self, config=None):
#         # if no config is supplied use the default roberta config (which is modified by adapter transformers.)
#         if config is None: config = RobertaConfig()
#         super(RobertaForCartography, self).__init__(config)
        
#     def forward(self, input_ids, attention_mask, labels=None,
#                 token_type_ids=None, inputs_embeds=None, **kwargs):
#         outputs = self.roberta(
#             input_ids=input_ids, 
#             attention_mask=attention_mask
#         )
#         sequence_output = outputs.last_hidden_state
#         logits = self.classifier(sequence_output)
#         # print(sequence_output.shape, logits.shape, self.num_labels)
#         output = {}
#         if labels is not None:
#             # single target class implies regression.
#             if self.num_labels == 1:
#                 criterion = nn.MSELoss()
#                 loss = criterion(logits.view(-1), labels.view(-1))
#             else:
#                 criterion = nn.CrossEntropyLoss()
#                 loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))
#             # return loss and logits if label is given
#             output["loss"] = loss
#         output["logits"] = logits
            
#         return output    
class RobertaWithAdapterForCartography(RobertaModelWithHeads):
    '''analogus to AdaptedRobertaForSequenceClassification. Includes support for adapters.'''
    def __init__(self, num_labels=2, task_name="mnli", config=None):
        # if no config is supplied use the default roberta config (which is modified by adapter transformers.)
        task_map = {
            "mnli": "multinli",
        }
        task_name = task_map.get(task_name, task_name)
        if config is None: config = RobertaConfig()
        super(RobertaWithAdapterForCartography, self).__init__(config)
        self.add_classification_head(task_name, num_labels=num_labels)
        self.adapter_task_name = task_name
        
    def activate(self):
        self.set_active_adapters(self.adapter_task_name)
        
    def forward(self, input_ids, attention_mask, labels=None,
                token_type_ids=None, inputs_embeds=None, **kwargs):
        outputs = self.roberta(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        # print(sequence_output.shape, logits.shape, self.num_labels)
        output = {}
        if labels is not None:
            # single target class implies regression.
            if self.num_labels == 1:
                criterion = nn.MSELoss()
                loss = criterion(logits.view(-1), labels.view(-1))
            else:
                criterion = nn.CrossEntropyLoss()
                loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))
            # return loss and logits if label is given
            output["loss"] = loss
        output["logits"] = logits
            
        return output