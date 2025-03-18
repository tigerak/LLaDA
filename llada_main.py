import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import json
from time import time
#torch
import torch
from torch.utils.data import DataLoader
# transformers
from transformers import AutoTokenizer
# modules
from config import *
from function.llada.llada_dataset import LLaDA_Dataset, collate_fn
from function.llada.llada_mask_predictor import LLaDA_MaskPredictor

class LLaDA_Config:
    def __init__(self, tokenizer):
        # 모델
        use_bf16 = True
        vocab_size = 32000
        type_vocab_size = 1

        hidden_dim = 768 # 2048
        num_attention_heads = 12 # 32
        head_dim = hidden_dim // num_attention_heads # 64
        dropout = 0.1

        intermediate_size = 3072 # 5634

        num_hidden_layers = 12 # 22
        # 코트나이저
        max_seq_len = 128
        pad_token_id = tokenizer.eos_token_id 
        eos_token_id = tokenizer.eos_token_id 
        # 학습
        epochs = 10
        # 추론론
        monte_carlo_samples = 10

        for key, value in locals().items():
            if key != "self":  # self 제외
                setattr(self, key, value)

        # (JSON 직렬화 방지)
        self.tokenizer = tokenizer

    def to_json(self):
        """JSON 변환 시 tokenizer 제외"""
        config_dict = {k: v for k, v in self.__dict__.items() if k != "tokenizer"}
        return json.dumps(config_dict, indent=4)

    def __repr__(self):
        return f"LLaDA_Config:\n{self.to_json()}"


if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained(KO_ROBERTA)

    cfg = LLaDA_Config(tokenizer)
    print(cfg)
    
    model = LLaDA_MaskPredictor(config=cfg)
    if cfg.use_bf16:
        model = model.to(dtype=torch.bfloat16)
    model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    dataset = LLaDA_Dataset(json_path=YTN_DATA, 
                            tokenizer=tokenizer,
                            config=cfg)
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            shuffle=True,
                            collate_fn=collate_fn)

    for epoch in range(cfg.epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            start_time = time()
            
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()
            t = batch["t"].cuda()

            batch_loss = 0
            logits, loss = model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 labels=labels,
                                 t=t)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            end_time = time()

            if (step+1) % 10 == 0:
                print(f"step:{step} / loss:{round(loss.item(), 4)} / time:{end_time-start_time}")
                predicted_tokens = torch.argmax(logits[0], dim=-1) 
                print(tokenizer.decode(predicted_tokens.tolist()))