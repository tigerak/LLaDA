import torch
import torch.nn.functional as F

def llada_inference(cfg,
                    model,
                    tokenizer,
                    prompt_ids: torch.Tensor,
                    num_step: int=8,
                    remask_str: str="low_confidence",
                    device="cuda"):
    model.eval()

    prompt_len = prompt_ids.size(1)
    total_len = cfg.max_seq_len

    prompt_ids = prompt_ids.unsqueeze(0).to(device)

    mask_token_id = tokenizer.mask_token_id

    init_input_ids = torch.full(
        (1, total_len),
        fill_value=mask_token_id,
        dtype=torch.long,
        device=device
    )
    init_input_ids[0, :prompt_len] = prompt_ids[0]

    attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)

    # denoisinf 스텝
    steps_t = torch.linspace(start=1.0, end=0.0, steps=num_step).tolist()

    current_ids = init_input_ids.clone()

    gen_token_len = 0

    for step_idx in range(num_step):
        t_val = steps_t[step_idx]

        with torch.no_grad():
            logits, _ = model(input_ids=current_ids,
                              attention_mask=None,
                              labels=None,
                              t=None)
            
        # 마스킹 된 위치에 대한 예측
        masked_positions = (current_ids == mask_token_id)
        if masked_positions.any():
            masked_logits = logits[masked_positions]

            predicted_ids = torch.argmax(masked_logits, dim=-1)

            current_ids[masked_positions] = predicted_ids
            #
            if gen_token_len == 0:
                gen_token_len = len(predicted_ids)

        # remask
        if step_idx < num_step - 1:
            if remask_str == "low_confidence":
                with torch.no_grad():
                    probs = F.softmax(masked_logits, dim=-1)
                    gather_conf = probs[range(len(predicted_ids)), predicted_ids]
                    sorted_conf, sorted_idx = torch.sort(gather_conf) # 오름차순 정렬
                    print(sorted_idx)
                    remask_ratio = (1.0 - (step_idx / num_step)) #* 0.5
                    print(remask_ratio)
                    # n_remask = int(remask_ratio * len(predicted_ids))
                    n_remask = int(remask_ratio * gen_token_len)
                    print("predicted_ids",gen_token_len)
                    print(n_remask)
                    ######################################################################
                    # -- n_remask 중 80%는 low-confidence, 20%는 random --
                    n_low = int(0.7 * n_remask)
                    n_rand = n_remask - n_low
                    # 1) low-confidence: sorted_idx[:n_low]
                    low_conf_idx = sorted_idx[:n_low]
                    # 2) random: sorted_idx[n_low:] 중 n_rand개
                    #    => 우선 'low-confidence로 이미 뽑힌 곳'을 제외하기 위해
                    remain_idx = sorted_idx[n_low:]  # 혹은 전체 predicted에서 중복 방지
                    if len(remain_idx) > 0 and n_rand > 7:
                        rand_perm = torch.randperm(remain_idx.size(0))[:n_rand]
                        rand_idx = remain_idx[rand_perm]
                    else:
                        rand_idx = torch.tensor([], dtype=torch.long, device=device)

                    # to_remask_idx = sorted_idx[:n_remask]
                    to_remask_idx = torch.cat([low_conf_idx, rand_idx], dim=0)
                    print(to_remask_idx)
                    ######################################################################
                    masked_pos_flat = masked_positions.view(-1).nonzero(as_tuple=True)[0]

                    global_remask_ids = masked_pos_flat[to_remask_idx]

                    current_ids.view(-1)[global_remask_ids] = mask_token_id

                    print(tokenizer.decode(current_ids.squeeze(0).tolist()))

            elif remask_str == "semi_autoregressive":
                
                pass
            else:
                pass

    
    return current_ids.squeeze(0)