import torch
import torch.nn.functional as F

def llada_inference(cfg,
                    model,
                    prompt_ids: torch.Tensor,
                    num_step: int=8,
                    remask_str: str="low_confidence",
                    device="cuda"):
    model.eval()

    prompt_len = prompt_ids.size(1)
    total_len = cfg.max_seq_len
    response_len = total_len - prompt_len

    prompt_ids = prompt_ids.unsqueeze(0).to(device)

    mask_token_id = cfg.mask_token_id

    init_input_ids = torch.full(
        (1, total_len),
        fill_value=mask_token_id,
        dtype=torch.long,
        device=device
    )
    init_input_ids[0, :prompt_len] = prompt_ids[0]

    attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)

    ### Denoising ###
    # 스텝 균등 분할
    steps_t = torch.linspace(start=1.0, end=0.0, steps=num_step + 1).tolist()
    
    current_ids = init_input_ids.clone()
    
    for step_idx in range(num_step):
        with torch.no_grad():
            torch.cuda.empty_cache() 
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

        # Remasking
        remask_ratio = steps_t[step_idx + 1]
        if step_idx + 1 < num_step: # 마지막 스텝 제외
            with torch.no_grad():
                probs = F.softmax(masked_logits, dim=-1)
                gather_conf = probs[range(len(predicted_ids)), predicted_ids]
                sorted_conf, sorted_idx = torch.sort(gather_conf) # 오름차순 정렬
                
                # remaskin 비율
                n_low = int(remask_ratio * response_len)
                n_rand = round(remask_ratio * (response_len - n_low))
                # 1) low-confidence: sorted_idx[:n_low]
                low_conf_idx = sorted_idx[:n_low]
                # 2) random: sorted_idx[n_low:] 중 n_rand개
                remain_idx = sorted_idx[n_low:] 
                if n_low > 10 and n_rand > 5:
                    rand_perm = torch.randperm(remain_idx.size(0))[:n_rand]
                    rand_idx = remain_idx[rand_perm]
                else:
                    rand_idx = torch.tensor([], dtype=torch.long, device=device)

                # to_remask_idx = sorted_idx[:n_remask]
                to_remask_idx = torch.cat([low_conf_idx, rand_idx], dim=0)
                ######################################################################
                masked_pos_flat = masked_positions.view(-1).nonzero(as_tuple=True)[0]

                global_remask_ids = masked_pos_flat[to_remask_idx]

                current_ids.view(-1)[global_remask_ids] = mask_token_id
            
        else : # 마지막 strp
            torch.cuda.empty_cache()
            return current_ids.squeeze(0).tolist()