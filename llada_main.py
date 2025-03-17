import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from time import time
#torch
import torch
from torch.utils.data import DataLoader
# transformers
from transformers import AutoTokenizer
# modules
from config import *
from function.llada.llada_dataset import LLaDA_Dataset
from function.llada.llada_mask_predictor import LLaDA_MaskPredictor

class LLaDA_Config:
    # 모델델
    use_bf16 = True

    vocab_size = 32000
    type_vocab_size = 1

    hidden_dim = 768 # 2048
    num_attention_heads = 12 # 32
    head_dim = hidden_dim // num_attention_heads
    dropout=0.1
    
    intermediate_size=3072 # 5634

    num_hidden_layers=12 # 22

    # 토크나이저
    max_seq_len = 128

    # 학습
    epochs = 10
    monte_carlo_samples = 10


if __name__ == "__main__":

    cfg = LLaDA_Config()
    print(cfg)

    tokenizer = AutoTokenizer.from_pretrained(KO_ROBERTA)

    model = LLaDA_MaskPredictor(config=cfg)
    if cfg.use_bf16:
        model = model.to(dtype=torch.bfloat16)
    model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    dataset = LLaDA_Dataset(json_path=YTN_DATA, 
                            tokenizer=tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            shuffle=True,
                            collate_fn=dataset.collate_fn)

    for epoch in range(cfg.epochs):
        model.train()
        for batch in dataloader:
            start_time = time()
            
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()
            t = batch["t"].cuda()

            batch_loss = 0
            logits, loss = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels,
                           t=t,
                           tokenizer=tokenizer)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            end_time = time()
