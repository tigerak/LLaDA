import torch

from transformers import AutoTokenizer

from config import *
from function.llada.llada_fnc import *
from function.llada.llada_transfoermer_block import LLaDA_BERT 

class LladaDecoderBlock(torch.nn.Module):
    def __init__(self, tokenizer):
        super(LladaDecoderBlock, self).__init__()
        
        self.tokenizer = tokenizer
        self.model = LLaDA_BERT()
        
    def train_model(self, 
                    input_batch, 
                    max_length):
        
        # tokenizing
        input = self.tokenizer(text=input_batch, 
                               padding="max_length",
                               max_length=max_length,
                               add_special_tokens=True, 
                               return_tensors="pt")
        masked_input_ids, mask_prob, t = mask_tokens_with_random_t(input["input_ids"], self.tokenizer)
        attention_mask = input["attention_mask"]
        
        # 라벨 생성성
        labels = input["input_ids"].clone()
        
    def forward(self, masked_input_ids, attention_mask, labels, t):
        # 모델을 통해 확률 분포 얻기
        output = self.model(input_ids=masked_input_ids, 
                            attention_mask=attention_mask)

        loss = calculate_loss(input_ids=masked_input_ids, 
                              logits=output, 
                              labels=labels, 
                              t=t, 
                              tokenizer=self.tokenizer)
        
        return dict(logits=output, 
                    loss=loss)
        

