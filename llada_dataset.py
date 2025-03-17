import json

# torch
import torch
from torch.utils.data import Dataset

class LLaDA_Dataset(Dataset):
    def __init__(self, json_path, tokenizer):

        with open(json_path, 'r', encoding="utf-8") as file:
            self.llada_data = json.load(file)

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.llada_data)
    
    def __getitem__(self, index):
        title = self.llada_data[index]["metadata"]["title"]
        article = self.llada_data[index]["document"]

        prompt_format = f"### TITLE: {title}\n### ARTICLE:"
        prompt_complete = f"{prompt_format} {article}"

        encoding = self.tokenizer(text=prompt_complete, 
                                  truncation=True,
                                  max_length=128,
                                  padding="max_length",
                                  add_special_tokens=True,
                                  return_tensors="pt")
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        prompt_enc = self.tokenizer(text=prompt_format, add_special_tokens=False)
        prompt_len = len(prompt_enc["input_ids"])

        masked_input_ids, mask_indices, t = mask_tokens_with_random_t(input_ids, 
                                                                   prompt_len, 
                                                                   self.tokenizer)

        labels = torch.full_like(input_ids, fill_value=-100)  # 처음부터 -100으로 초기화
        labels[mask_indices] = input_ids[mask_indices]  # 마스킹된 부분만 정답으로 사용
        
        return {"input_ids": masked_input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "t": torch.Tensor(t)}
    
@staticmethod
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    t = torch.tensor([item["t"] for item in batch])

    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "t": t}

def mask_tokens_with_random_t(input_ids, 
                              prompt_len,
                              tokenizer,
                              max_t=1.0,
                              min_t=0.1):
    """
    - `t`를 [min_t, max_t] 범위에서 랜덤하게 샘플링하여 각 샘플마다 다른 마스킹 비율 적용
    """

    seq_len = input_ids.shape[0]
    
    # Monte Carlo 방식으로 `t`를 샘플링
    t = torch.rand(1).item() * (max_t - min_t) + min_t  # 0.1~1.0 사이 랜덤한 float 값

    # 🔹 프롬프트 & 특수 토큰 제외한 마스킹 가능한 토큰 찾기
    valid_tokens = torch.ones(seq_len, dtype=torch.bool)  # 모든 위치를 True로 초기화
    valid_tokens[:prompt_len] = False  # 프롬프트 제외
    valid_tokens[input_ids == tokenizer.pad_token_id] = False  # 패딩 제외
    valid_tokens[input_ids == tokenizer.bos_token_id] = False  # BOS 제외

    # `t` 확률로 마스킹 여부 결정
    valid_indices = valid_tokens.nonzero(as_tuple=True)[0]  # 마스킹 가능한 인덱스
    num_to_mask = max(1, round(t * valid_indices.shape[0]))  # 최소 1개 이상 마스킹

    # num_to_mask 개수만큼 선택하여 마스킹
    mask_indices = valid_indices[torch.randperm(len(valid_indices))[:num_to_mask]]

    # 마스킹 적용
    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_indices] = tokenizer.mask_token_id  # `[MASK]` 적용
    
    return masked_input_ids, mask_indices, t
    