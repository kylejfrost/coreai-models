# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Qwen3.5 MoE macOS authoring.

This model family combines the Qwen3.5 hybrid linear/full attention decoder
with Qwen-style top-k MoE feed-forward blocks. Text weights in the multimodal
checkpoints live under ``model.language_model.``; vision and MTP weights are
dropped for text-only export.
"""

import re

import torch
import torch.nn as nn
from transformers import AutoConfig, PretrainedConfig
from typing_extensions import Self, override

from coreai_models.models.base import BaseForCausalLM
from coreai_models.models.macos.qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNorm,
    Qwen3_5TextConfig,
    _is_full_layer,
    _register_hf as _register_qwen3_5_hf,
)
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.switch import SwitchGLU


class Qwen3_5MoeTextConfig(Qwen3_5TextConfig):
    """Text-decoder config for ``qwen3_5_moe_text``."""

    model_type = "qwen3_5_moe_text"

    def __init__(
        self,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        num_experts_per_tok: int = 8,
        num_experts: int = 256,
        norm_topk_prob: bool | None = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        intermediate_size: int | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("hidden_size", 2048)
        kwargs.setdefault("num_hidden_layers", 40)
        kwargs.setdefault("num_attention_heads", 16)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("max_position_embeddings", 32768)
        kwargs.setdefault("linear_num_key_heads", 16)
        kwargs.setdefault("linear_num_value_heads", 32)
        kwargs.setdefault("linear_key_head_dim", 128)
        kwargs.setdefault("linear_value_head_dim", 128)
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = True if norm_topk_prob is None else norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        super().__init__(
            intermediate_size=intermediate_size or moe_intermediate_size,
            **kwargs,
        )


class Qwen3_5MoeConfig(PretrainedConfig):
    """Top-level config for multimodal ``qwen3_5_moe`` checkpoints."""

    model_type = "qwen3_5_moe"
    sub_configs = {"text_config": Qwen3_5MoeTextConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = Qwen3_5MoeTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3_5MoeTextConfig()
        self.text_config = text_config
        self.vision_config = vision_config
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


def _register_hf() -> None:
    """Register Qwen3.5 MoE configs with ``AutoConfig``."""
    _register_qwen3_5_hf()
    for model_type, cfg_cls in (
        ("qwen3_5_moe", Qwen3_5MoeConfig),
        ("qwen3_5_moe_text", Qwen3_5MoeTextConfig),
    ):
        try:
            AutoConfig.register(model_type, cfg_cls)
        except ValueError:
            pass


_register_hf()


class Qwen3_5SparseMoeBlock(nn.Module):
    """Qwen3.5 MoE block with packed ``SwitchGLU`` expert weights."""

    def __init__(self, config) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
        self.gate = nn.Linear(hidden, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            hidden,
            config.moe_intermediate_size,
            config.num_experts,
            bias=False,
        )

        shared_size = int(getattr(config, "shared_expert_intermediate_size", 0) or 0)
        if shared_size > 0:
            self.shared_expert = MLP(hidden, shared_size, bias=False)
            self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.softmax(self.gate(x), dim=-1, dtype=torch.float32)
        scores, indices = torch.topk(gates, self.top_k, dim=-1, largest=True)
        if self.norm_topk_prob:
            scores = scores / torch.sum(scores, dim=-1, keepdim=True)

        y = self.switch_mlp(x, indices.to(torch.uint16))
        y = torch.sum(y * scores.unsqueeze(-1).to(y.dtype), dim=-2)

        if hasattr(self, "shared_expert"):
            shared = self.shared_expert(x)
            shared = torch.sigmoid(self.shared_expert_gate(x)) * shared
            y = y + shared

        return y.to(x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.is_full = _is_full_layer(config, layer_idx)
        if self.is_full:
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        self.mlp = Qwen3_5SparseMoeBlock(config)
        self.input_layernorm = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = self.input_layernorm(x)
        if self.is_full:
            h = self.self_attn(h, position_ids, cache)
        else:
            h = self.linear_attn(h, cache)
        x = x + h
        r = self.mlp(self.post_attention_layernorm(x))
        return x + r


class Qwen3_5MoeModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        return self.norm(h)


class Qwen3_5MoeForCausalLM(BaseForCausalLM):
    """Qwen3.5 MoE text decoder (``qwen3_5_moe``)."""

    _HF_MODEL_CLASS = None

    @override
    def _init_model(self, config) -> None:
        self.model = Qwen3_5MoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
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

    @staticmethod
    def _is_text_key(key: str) -> bool:
        return not (
            key.startswith(("model.visual.", "visual.", "model.mtp", "mtp."))
            or ".visual." in key
        )

    @classmethod
    def _prepare_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> None:
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not cls._is_text_key(key):
                continue
            if key.startswith("model.language_model."):
                key = "model." + key[len("model.language_model.") :]
            if key.endswith(".linear_attn.conv1d.weight") and value.ndim == 2:
                value = value.unsqueeze(1)
            remapped[key] = value

        state_dict.clear()
        state_dict.update(remapped)

        moe_layers = sorted(
            {
                int(m.group(1))
                for key in state_dict
                if (m := re.match(r"model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$", key))
            }
        )
        for layer_idx in moe_layers:
            prefix = f"model.layers.{layer_idx}.mlp."
            gate_up = state_dict.pop(prefix + "experts.gate_up_proj")
            mid = gate_up.shape[-2] // 2
            state_dict[prefix + "switch_mlp.gate_proj.weight"] = (
                gate_up[:, :mid, :].unsqueeze(0).contiguous()
            )
            state_dict[prefix + "switch_mlp.up_proj.weight"] = (
                gate_up[:, mid:, :].unsqueeze(0).contiguous()
            )
            state_dict[prefix + "switch_mlp.down_proj.weight"] = (
                state_dict.pop(prefix + "experts.down_proj").unsqueeze(0).contiguous()
            )

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        self._prepare_state_dict(state_dict)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        remapped = dict(state_dict)
        self._prepare_state_dict(remapped)
        super().load_state_dict(remapped, strict=strict, assign=assign)
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
