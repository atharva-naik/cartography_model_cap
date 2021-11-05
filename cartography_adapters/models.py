from transformers import (BertConfig,
                          RobertaConfig,
                          BertForSequenceClassification, 
                          RobertaForSequenceClassification)


class BertForCartography(BertForSequenceClassification):
    '''analogus to AdaptedBertForSequenceClassification. Includes support for adapters.'''
    def __init__(self, config=None):
        # if no config is supplied use the default roberta config (which is modified by adapter transformers.)
        if config is None: config = BertConfig()
        super(BertForCartography, self).__init__(config)    


class RobertaForCartography(RobertaForSequenceClassification):
    '''analogus to AdaptedRobertaForSequenceClassification. Includes support for adapters.'''
    def __init__(self, config=None):
        # if no config is supplied use the default roberta config (which is modified by adapter transformers.)
        if config is None: config = RobertaConfig()
        super(RobertaForCartography, self).__init__(config)
        
    def forward(self, input_ids, attention_mask, labels=None,
                token_type_ids=None, inputs_embeds=None, **kwargs):
        outputs = self.roberta(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        print(sequence_output.shape, logits.shape, self.num_labels)
        output = {}
        if labels is not None:
            # single target class implies regression.
            if self.num_labels == 1:
                criterion = MSELoss()
                loss = criterion(logits.view(-1), labels.view(-1))
            else:
                criterion = CrossEntropyLoss()
                loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))
            # return loss and logits if label is given
            output["loss"] = loss
        output["logits"] = logits
            
        return output