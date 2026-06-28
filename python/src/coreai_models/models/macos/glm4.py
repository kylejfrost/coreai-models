# GLM-4 (dense, `glm4` / Glm4ForCausalLM — the GLM-4-*-0414 series) authored for Core AI.
#
# Differences from the qwen3 reference this is modeled on (all verified against transformers
# `models/glm4/modeling_glm4.py`):
#   * Attention uses SEPARATE q/k/v projections WITH bias (`attention_bias`, default True),
#     no q/k RMSNorm, and **partial interleaved** RoPE (`partial_rotary_factor`, default 0.5 →
#     only the first head_dim*0.5 dims are rotated, GPT-J/interleaved pairing 2j<->2j+1).
#     Apple's composite RoPE reproduces this exactly with `dims=rotary_dim, interleaved=True`.
#   * Decoder block uses Gemma2-style **sandwich norms**: input_layernorm before attn +
#     post_self_attn_layernorm after attn (pre-residual); post_attention_layernorm before mlp +
#     post_mlp_layernorm after mlp (pre-residual).
#   * MLP is SwiGLU with a FUSED gate_up_proj (split into gate/up at load).
import re

import torch
import torch.nn as nn
from transformers import Glm4Config
from transformers import Glm4ForCausalLM as HFGlm4ForCausalLM
from typing_extensions import Self, override

from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA


class Attention(nn.Module):
    def __init__(self, config: Glm4Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", dim // n_heads)
        bias = bool(getattr(config, "attention_bias", True))

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.sdpa = SDPA(is_causal=True)
        prf = float(getattr(config, "partial_rotary_factor", 0.5))
        rotary_dim = int(head_dim * prf)
        base = float(getattr(config, "rope_theta", 10000.0))
        # partial (dims=rotary_dim) + interleaved (GPT-J pairing) == GLM-4's rotary
        self.rope = RoPE(base=base, dims=rotary_dim, interleaved=True)

    def forward(
        self, x: torch.Tensor, position_ids: torch.IntTensor, cache: KVCache | None = None
    ) -> torch.Tensor:
        b, q_len, _ = x.shape
        q = self.q_proj(x).reshape(b, q_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, q_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, q_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(q_len)
        torch._check_is_size(seq_len)
        offset = seq_len - q_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, q_len)
        q = self.rope(q, position_ids=rope_positions)
        k = self.rope(k, position_ids=rope_positions)

        if cache is not None:
            k, v = cache.update_and_fetch(
                self.layer_idx, offset, k, v, seq_len=seq_len, query_len=q_len
            )
        out = (
            self.sdpa(q, k, v)
            .permute(0, 2, 1, 3)
            .reshape(b, q_len, self.n_heads * self.head_dim)
        )
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: Glm4Config, layer_idx: int) -> None:
        super().__init__()
        h = config.hidden_size
        eps = config.rms_norm_eps
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(h, config.intermediate_size)
        self.input_layernorm = RMSNorm(h, eps=eps)
        self.post_self_attn_layernorm = RMSNorm(h, eps=eps)
        self.post_attention_layernorm = RMSNorm(h, eps=eps)
        self.post_mlp_layernorm = RMSNorm(h, eps=eps)

    def forward(
        self, x: torch.Tensor, position_ids: torch.IntTensor, cache: KVCache | None = None
    ) -> torch.Tensor:
        # GLM-4 sandwich norms (post-norm applied BEFORE the residual add)
        r = self.self_attn(self.input_layernorm(x), position_ids, cache)
        r = self.post_self_attn_layernorm(r)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        r = self.post_mlp_layernorm(r)
        return h + r


class Glm4Model(nn.Module):
    def __init__(self, config: Glm4Config) -> None:
        super().__init__()
        h = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, h)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(h, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        hs = self.embed_tokens(input_ids)
        for layer in self.layers:
            hs = layer(hs, position_ids, cache)
        return self.norm(hs)


class Glm4ForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFGlm4ForCausalLM

    @override
    def _init_model(self, config: Glm4Config) -> None:
        self.model = Glm4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(input_ids, position_ids, cache)
        return self.lm_head(out)

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Split GLM's fused gate_up_proj [2*intermediate, hidden] -> gate_proj + up_proj.
        # transformers Glm4MLP: gate, up = gate_up(x).chunk(2); out = down(up * silu(gate))
        # => first half is gate, second half is up (matches Core AI MLP: down(up * silu(gate))).
        layers = {
            int(m.group(1))
            for k in state_dict
            if (m := re.match(r"model\.layers\.(\d+)\.mlp\.gate_up_proj\.weight$", k))
        }
        for i in layers:
            w = state_dict.pop(f"model.layers.{i}.mlp.gate_up_proj.weight")
            inter = w.shape[0] // 2
            state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = w[:inter].clone()
            state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = w[inter:].clone()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
