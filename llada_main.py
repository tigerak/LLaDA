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
from function.llada.llada_inference import llada_inference

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
        mask_token_id = tokenizer.mask_token_id
        # 학습
        batch = 32
        lr = 2e-5
        epochs = 50
        # 추론론
        monte_carlo_samples = 32

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    ### 기존 학습 모델 불러오기 ###
    leaning_model_path = save_dir + r'llada/last_model.pt'
    start_epoch = 0
    if os.path.exists(leaning_model_path):
        print(f"기존 학습 모델 로드 시작: {leaning_model_path}")
        checkpoint = torch.load(leaning_model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = cfg.lr
            print(f"새로운 학습률 적용: {optimizer.param_groups[0]['lr']}")
        start_epoch = checkpoint["epoch"] 
        print(f"모델 로드 완료! {start_epoch}epoch 부터 학습을 재개합니다.")
    else:
        print(f"저장된 모델이 없습니다. 새 모델을 학습합니다.")

    model.cuda()

    # (4) 옵티마이저 state도 GPU로 올림 (float32 유지 or bf16 변환은 상황에 맞게)
    for param_state in optimizer.state.values():
        if isinstance(param_state, dict):
            for k, v in param_state.items():
                if torch.is_tensor(v):
                    # 일단 float32 그대로 GPU로만 옮기는 것이 일반적
                    param_state[k] = v.cuda() # .to(torch.bfloat16)
    
    def get_model_size(model):
        total_params = sum(p.numel() for p in model.parameters())
        total_memory = total_params * 2 / (1024 ** 2)  # MB 단위 (FP16 기준)
        print(f"총 파라미터 수: {total_params:,}")
        print(f"메모리 사용량 (FP16 기준): {total_memory:.2f} MB")
    get_model_size(model)
    
    #### 데이터 불러오기 ###
    dataset = LLaDA_Dataset(json_path=YTN_DATA, 
                            tokenizer=tokenizer,
                            config=cfg)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.batch,
                            shuffle=True,
                            collate_fn=collate_fn)

    ### 학습 시작 ###
    print(f"에폭 당 스텝:{len(dataloader)}")
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        # for step, batch in enumerate(dataloader):
        #     start_time = time()
            
        #     input_ids = batch["input_ids"].cuda()
        #     attention_mask = batch["attention_mask"].cuda()
        #     labels = batch["labels"].cuda()
        #     t = batch["t"].cuda()

        #     batch_loss = 0
        #     logits, loss = model(input_ids=input_ids,
        #                          attention_mask=None,
        #                          labels=labels,
        #                          t=t)
            
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()

        #     end_time = time()

        #     if (step+1) % 100 == 0:
        #         print(f"step:{step} / epoch:{epoch} / loss:{round(loss.item(), 4)} / time:{round(end_time-start_time, 2)} / t:{round(t.tolist()[0], 2)}")
        #         predicted_tokens = torch.argmax(logits[0], dim=-1) 
        #         print(tokenizer.decode(predicted_tokens.tolist()))

        #     # if step == 1:
        #     #     break

        ### inference test ##
        # test_title = "### TITLE: 한전, 지난해 영업익 8조 3천억...4년 만에 흑자 전환\n###ARTICLE: "
        test_title = "### TITLE: 작년 역대급 실적 SK하이닉스, 차입금 6조7천 억 상환\n###ARTICLE: "
        encoding = tokenizer(text=test_title, 
                            add_special_tokens=False,
                            return_tensors="pt").to("cuda") 
        prompt_ids = encoding["input_ids"]
        output = llada_inference(cfg=cfg,
                                 model=model,
                                 prompt_ids=prompt_ids,
                                 num_step=cfg.monte_carlo_samples,
                                 remask_str="low_confidence",
                                 device="cuda")
        print(tokenizer.decode(output))
        break
        
        # 모델 저장 (epoch마다 저장)
        model_save_path = os.path.join(save_dir, f"llada/last_model.pt")
        # model_save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, model_save_path)
        
        print(f"모델 저장 완료: {model_save_path}")