import torch
import torch.nn as nn
# module
from function.llada.llada_pred_module import LLaDA_Pred_Module
from function.llada.llada_fnc import calculate_loss


class LLaDA_MaskPredictor(nn.Module):
    def __init__(self, config):
        super(LLaDA_MaskPredictor, self).__init__()
        self.model = LLaDA_Pred_Module(config=config)
        
    def forward(self, 
                input_ids,
                attention_mask,
                labels,
                t,
                tokenizer):
        logits = self.model(input_ids=input_ids,
                            attention_mask=attention_mask)
        loss = None
        if labels is not None:
            loss = calculate_loss(input_ids=input_ids,
                                  logits=logits,
                                  labels=labels,
                                  t=t,
                                  tokenizer=tokenizer)
            
        return logits, loss