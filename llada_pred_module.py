import torch
import torch.nn as nn
# module
from function.llada.llada_embedding_layer import LLaDA_EmbeddingLayer
from function.llada.llada_transfoermer_block import LLaDA_TransformerBlock
from function.rope.rope_fnc import build_rope_cache

class LLaDA_Pred_Module(nn.Module):
    def __init__(self, config):
        super(LLaDA_Pred_Module, self).__init__()
        self.config = config
        # 임베딩
        self.embed = LLaDA_EmbeddingLayer(config)
        # RoPE 사전 계산
        rope_cos, rope_sin = build_rope_cache(seq_len=config.max_seq_len,
                                              head_dim=config.head_dim)
        # register_buffer로 올려놓으면 학습 안 되는 파라미터로 고정
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self.blocks = nn.ModuleList(
            [LLaDA_TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

        self.rn_final = nn.RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=True)
        
    def forward(self, input_ids, attention_mask):
        hidden_states = self.embed(input_ids)

        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states=hidden_states,
                                  attention_mask=attention_mask,
                                  rope_cache=(self.rope_cos, self.rope_sin),
                                  offset=0)
        norm_out = self.rn_final(hidden_states)    
        logits = self.lm_head(norm_out)
        return logits