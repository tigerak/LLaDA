import torch.nn as nn

class LLaDA_EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 실제 vocab_size가 맞는지 확인 (tokenizer.vocab_size 등)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)

        # RoPE 사용 시, 별도의 position embedding은 없어도 괜찮음
        # (LLaMA style: rotary pos emb)
        # 필요하다면 nn.Embedding(config.max_seq_len, config.hidden_dim) 추가 가능

        # Optionally layer norm 등
        # self.norm = nn.RMSNorm(config.hidden_dim)

    def forward(self, input_ids):
        # input_ids: (B, S) int
        # -> (B, S, d)
        x = self.token_embeddings(input_ids)
        # x = self.norm(x)  # 필요하면
        return x