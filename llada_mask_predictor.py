import torch
import torch.nn as nn
# module
from function.llada.llada_transfoermer_block import LLaDA_TransformerBlock
from function.rope.rope_fnc import build_rope_cache

class LLaDA_MaskPredictor(nn.modules):
    """
    논문의 'mask predictor'를 구현한 모듈.
    입력: (B, L) shape의 input_ids (마스킹된 형태)
    출력: (B, L, vocab_size)의 로짓.
    """
    def __init__(self, config):
        super(LLaDA_MaskPredictor, self).__init__()
        self.config = config
        self.block = nn.ModuleList(
            [LLaDA_TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_final = nn.RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
            return output