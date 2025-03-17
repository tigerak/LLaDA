import math
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# module
from function.rope.rope_fnc import apply_rotary_pos_emb

class LLaDA_TransformerBlock(nn.Module):
    def __init__(self, config):
        super(LLaDA_TransformerBlock, self).__init__()
        self.hidden_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = config.dropout

        self.ln1 = nn.RMSNorm(config.hidden_size, eps=1e-5)
        
        self.q_peoj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_peoj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_peoj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.o_peoj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.ln2 = nn.RMSNorm(config.hidden_size, eps=1e-5)

        self.ffn = nn.Sequential(
                        nn.Linear(config.hidden_size, config.intermediate_size*2, bias=False),
                        SwiGLU(config.intermediate_size*2, config.intermediate_size),
                        nn.Linear(config.intermediate_size, config.hidden_size, bias=False) 
                        )

    def forward(self, hidden_states, attention_mask, rope_cache=None, offset=0):
        B, S = hidden_states.size()

        attn_in = self.ln1(hidden_states)

        q = self.q_peoj(attn_in)
        k = self.k_peoj(attn_in)
        v = self.v_peoj(attn_in) # (B, S, d)

        q = q.view(B, S, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim)

        if rope_cache is not None:
            q, k = apply_rotary_pos_emb(q=q,
                                        k=k,
                                        rope_cache=rope_cache,
                                        offset=offset)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3) # (B, n_head, S, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 2: # (B, S) -> (B, 1, 1, S)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores + attention_mask

        attn_props = F.softmax(attn_scores, dim=-1)
        attn_props = F.dropout(input=attn_props,
                               p=self.dropout,
                               training=self.training)
        attn_out = torch.matmul(attn_props, v) # (B, n_head, S, head_dim)

        attn_out = attn_out.permute(0, 2, 1, 3).contiguous() # (B, S, n_head, head_dim)
        attn_out = attn_out.view(B, S, self.hidden_dim) # (B, S, d)

        attn_out = self.o_peoj(attn_out)

        residual1 = hidden_states + attn_out 

        ffn_in = self.ln2(residual1)
        ffn_out = self.ffn(ffn_in)

        residual2 = residual1 + ffn_out

        return residual2
    
class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=True)
        self.fc2 = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        return F.silu(self.fc1(x)) * self.fc2(x)
    

