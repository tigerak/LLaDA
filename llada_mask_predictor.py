import torch
import torch.nn as nn
# module
from function.llada.llada_pred_module import LLaDA_Pred_Module
# from function.llada.llada_fnc import calculate_loss


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
            loss = self.calculate_loss(logits=logits,
                                       labels=labels)
            
        return logits, loss
    
    def calculate_loss(self, logits, labels):
    
        # # 마스크된 위치 찾기
        # mask_positions = (input_ids == tokenizer.mask_token_id)
        
        # 손실 계산을 위한 CrossEntropyLoss 함수
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        # 원본 logits에서 마스킹된 부분만 손실 계산하도록 조정
        vocab_size = logits.shape[-1]
        loss = loss_fn(logits.view(-1, vocab_size), 
                    labels.view(-1))  # (B*Seq_len)

        # # 마스킹된 토큰만 손실 유지, 나머지는 무시
        # loss = loss.view(input_ids.shape)  # 원래 크기로 reshape
        # loss = loss[mask_positions]  # 마스크된 부분만 선택

        return loss.mean()  # 배치 평균