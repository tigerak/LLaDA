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
                t):
        logits = self.model(input_ids=input_ids,
                            attention_mask=attention_mask)
        loss = None
        if labels is not None:
            loss = self.calculate_loss(logits=logits,
                                       labels=labels,
                                       t=t)
            
        return logits, loss
    
    def calculate_loss(self, logits, labels, t):
        """
          - (B, L, V) 로짓 & (B, L) 레이블
          - CrossEntropy(reduction='none')로 모든 토큰 위치별 손실값 계산
          - 마스킹되지 않은 위치(-100)는 무시
          - 각 샘플별 마스킹 토큰 평균 후 1/t 곱
          - 배치 평균
        """
        # 손실 계산을 위한 CrossEntropyLoss 함수
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        B, L, vocab_size = logits.shape
        # => logits: (B,L,vocab_size) -> (B*L, vocab_size)
        # => labels: (B,L) -> (B*L,)
        ce_1d = loss_fn(logits.view(-1, vocab_size), labels.view(-1))  # shape (B*L)

        # 2) 2D로 reshape (B,L)
        ce_2d = ce_1d.view(B, L)  # (B, L)

        # 3) (마스킹된) 유효 토큰만 평균 내고, 1/t 곱
        #    ignore_index=-100인 곳은 이미 CE가 0 혹은 무시 처리가 됨
        #    (PyTorch 내부적으로 "reduction=none + ignore_index" = 그 위치 0)
        #    그러나 "평균"은 직접 계산해야 하므로 아래와 같이 구현
        losses_per_sample = []

        for i in range(B):
            # (i번 샘플에서) 어느 토큰이 실제 마스킹되어 학습 대상인가?
            #  => labels[i,j] != -100  => 그 위치만 CE > 0
            valid_mask = (labels[i] != -100)  # shape (L,)

            ce_valid = ce_2d[i, valid_mask]   # 해당 샘플에서 마스킹된 위치들의 CE
            if ce_valid.numel() == 0:
                # # 혹시 마스킹된 토큰이 하나도 없으면(=prompt만 있는 경우 등),
                # # 그냥 0 손실로 처리하거나, 다른 예시에만 평균 반영할 수도 있음
                # losses_per_sample.append(0.0)

                # 샘플 i에서 마스킹된 토큰이 전혀 없으면
                # 0.0을 PyTorch 텐서로 만들어 추가 (requires_grad=False 문제?)
                # → "0.0"을 그냥 배치 내 다른 텐서와 합산시킴
                #    혹은 별도 처리(continue) 가능
                zero_t = ce_2d.new_zeros(1)  # ce_2d와 동일 dtype/device
                losses_per_sample.append(zero_t)
                continue
            
            mean_ce_i = ce_valid.mean()  # (마스킹 위치) 평균 CE
            # 1/t 곱하여 서로 다른 t들에 대해 스케일링링
            # scaled_loss_i = mean_ce_i * (1.0 / t[i].item())
            scaled_loss_i = mean_ce_i * (1.0 / t[i])   # 텐서 (requires_grad=True)
            
            losses_per_sample.append(scaled_loss_i)

        # 4) 배치 평균
        # final_loss = torch.stack([torch.tensor(v, device=logits.device, dtype=logits.dtype)
        #                           for v in losses_per_sample]).mean()
        final_loss = torch.stack(losses_per_sample).mean()  # 여기선 tensor(...) 재생성 없음
        
        return final_loss