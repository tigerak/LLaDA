import torch
import torch.nn.functional as F

def calculate_loss(input_ids, logits, labels, t, tokenizer):
    
    # 마스크된 위치 찾기
    mask_positions = (input_ids == tokenizer.mask_token_id)
    
    # 손실 계산을 위한 CrossEntropyLoss 함수
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    # 원본 logits에서 마스킹된 부분만 손실 계산하도록 조정
    vocab_size = logits.shape[-1]
    loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))  # (B*Seq_len)

    # 마스킹된 토큰만 손실 유지, 나머지는 무시
    loss = loss.view(input_ids.shape)  # 원래 크기로 reshape
    loss = loss[mask_positions]  # 마스크된 부분만 선택
    
    return loss.mean()  # 배치 평균

# def mask_tokens_with_random_t(input_ids, tokenizer):
#     """
#     - `t`를 [min_t, max_t] 범위에서 랜덤하게 샘플링하여 각 샘플마다 다른 마스킹 비율 적용
#     """
#     max_t = 1.0   
#     min_t = 0.1

#     batch_size, seq_len = input_ids.shape
    
#     # Monte Carlo 방식으로 `t`를 샘플링
#     # 어떻게 해야 더 골고루 샘플링할 수 있을지 고민해볼 것
#     t = torch.rand(batch_size, 1) * (max_t - min_t) + min_t  # (batch_size, 1)

#     # `t` 확률로 마스킹 여부 결정
#     mask_prob = torch.rand(batch_size, seq_len) < t

#     special_tokens_mask = (input_ids == tokenizer.pad_token_id) | \
#                           (input_ids == tokenizer.bos_token_id) # eos는 예측 가능하여야 함
#     mask_prob[special_tokens_mask] = False  # 특수 토큰은 마스킹하지 않음

#     masked_input_ids = input_ids.clone()
#     masked_input_ids[mask_prob] = tokenizer.mask_token_id
    
#     return masked_input_ids, t
    