# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Gemma-4 macOS authoring.

Implements the Gemma-4 text decoder (``gemma4``) and the EAGLE-style draft
(``gemma4_assistant``) for the Core AI macOS export path. Gemma-4 is not yet
in ``transformers``, so this module also registers lightweight HuggingFace
``PretrainedConfig`` classes with ``AutoConfig`` (the export pipeline calls
``AutoConfig.from_pretrained`` before our model registry is consulted) and
installs a narrow shim so ``AutoTokenizer`` can load the Gemma-4 tokenizer.

Gemma-4 vs Gemma-3 deltas handled here:

* model_type ``gemma4`` (multimodal ``Gemma4ForConditionalGeneration``; text
  decoder lives under the ``model.language_model.`` safetensors prefix) and
  ``gemma4_assistant`` (standalone draft, keys under ``model.``).
* 5:1 sliding:full attention, ``sliding_window`` 1024 (``layer_types``).
* Dual RoPE: full/global layers use theta 1e6 with ``partial_rotary_factor``
  0.25 (partial rotary); sliding layers use theta 1e4, full rotary.
* ``head_dim`` 256 for sliding, ``global_head_dim`` 512 for full/global layers,
  with 16 vs 4 KV heads -> the per-layer KV width is non-uniform, so K/V are
  packed into / unpacked from the uniform export KV-cache tensor.
* ``attention_k_eq_v`` -> full/global layers share one projection for K and V
  (no ``v_proj`` weight).
* ``final_logit_softcapping`` 30.0 (full model; ``None`` for the assistant),
  gemma RMSNorm(+1), tied embeddings, vocab 262144.

The assistant is EAGLE-style: it consumes a backbone hidden state and shares
the backbone KV cache, so it is *not* a faithful standalone decoder. See
``Gemma4AssistantForCausalLM`` for the exported-decoder approximation and the
documented gap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, PretrainedConfig
from typing_extensions import Self, override

from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.switch import SwitchGLU
# Gemma-4 RMSNorm uses a plain multiplicative weight (NOT the Gemma-3 (1+weight)
# convention): verified against the reference modeling code and the checkpoint
# weights (norm means ~1-7, with negative entries — i.e. full weights, not ~0).
from coreai_models.primitives.macos.rms_norm import RMSNorm as Gemma4RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA

# --------------------------------------------------------------------------- #
# HuggingFace config classes (Gemma-4 is not in transformers yet)
# --------------------------------------------------------------------------- #


class Gemma4TextConfig(PretrainedConfig):
    """Text-decoder config shared by ``gemma4`` and ``gemma4_assistant``."""

    model_type = "gemma4_text"

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 5376,
        intermediate_size: int = 21504,
        num_hidden_layers: int = 60,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 16,
        num_global_key_value_heads: int = 4,
        head_dim: int = 256,
        global_head_dim: int = 512,
        sliding_window: int = 1024,
        layer_types: list[str] | None = None,
        rope_parameters: dict | None = None,
        rms_norm_eps: float = 1e-6,
        final_logit_softcapping: float | None = None,
        attention_k_eq_v: bool = True,
        max_position_embeddings: int = 262144,
        hidden_activation: str = "gelu_pytorch_tanh",
        enable_moe_block: bool = False,
        num_experts: int = 0,
        top_k_experts: int = 8,
        moe_intermediate_size: int = 0,
        pad_token_id: int = 0,
        tie_word_embeddings: bool = True,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_global_key_value_heads = num_global_key_value_heads
        self.head_dim = head_dim
        self.global_head_dim = global_head_dim
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.rope_parameters = rope_parameters
        self.rms_norm_eps = rms_norm_eps
        self.final_logit_softcapping = final_logit_softcapping
        self.attention_k_eq_v = attention_k_eq_v
        self.max_position_embeddings = max_position_embeddings
        self.hidden_activation = hidden_activation
        self.enable_moe_block = enable_moe_block
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.moe_intermediate_size = moe_intermediate_size
        super().__init__(
            pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )


class Gemma4Config(PretrainedConfig):
    """Top-level config for the multimodal ``gemma4`` checkpoint."""

    model_type = "gemma4"
    sub_configs = {"text_config": Gemma4TextConfig}

    def __init__(self, text_config=None, tie_word_embeddings: bool = True, **kwargs) -> None:
        if isinstance(text_config, dict):
            text_config = Gemma4TextConfig(**text_config)
        elif text_config is None:
            text_config = Gemma4TextConfig()
        self.text_config = text_config
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Gemma4AssistantConfig(PretrainedConfig):
    """Top-level config for the EAGLE-style ``gemma4_assistant`` draft."""

    model_type = "gemma4_assistant"
    sub_configs = {"text_config": Gemma4TextConfig}

    def __init__(
        self,
        text_config=None,
        backbone_hidden_size: int = 5376,
        tie_word_embeddings: bool = True,
        **kwargs,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = Gemma4TextConfig(**text_config)
        elif text_config is None:
            text_config = Gemma4TextConfig()
        self.text_config = text_config
        self.backbone_hidden_size = backbone_hidden_size
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


def _register_hf() -> None:
    """Register Gemma-4 configs with AutoConfig and shim the tokenizer.

    Idempotent: safe to call from multiple import paths.
    """
    for model_type, cfg_cls in (
        ("gemma4", Gemma4Config),
        ("gemma4_assistant", Gemma4AssistantConfig),
        ("gemma4_text", Gemma4TextConfig),
    ):
        try:
            AutoConfig.register(model_type, cfg_cls)
        except ValueError:
            # Already registered (e.g. re-import) — fine.
            pass

    # The Gemma-4 tokenizer_config.json stores `extra_special_tokens` as a list,
    # but transformers 4.57's GemmaTokenizerFast expects a dict and crashes in
    # `_set_model_specific_special_tokens`. Coerce list -> dict so AutoTokenizer
    # (used by the bundle writer + quant calibration) can load it. Narrow: only
    # changes behavior for the malformed list case.
    import transformers.tokenization_utils_base as _tub

    _method = "_set_model_specific_special_tokens"
    _orig = getattr(_tub.PreTrainedTokenizerBase, _method)
    if not getattr(_orig, "_gemma4_patched", False):

        def _patched(self, special_tokens):  # type: ignore[no-untyped-def]
            if isinstance(special_tokens, list):
                special_tokens = {
                    t: t for t in special_tokens if isinstance(t, str)
                }
            return _orig(self, special_tokens)

        _patched._gemma4_patched = True  # type: ignore[attr-defined]
        setattr(_tub.PreTrainedTokenizerBase, _method, _patched)


_register_hf()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _is_global_layer(config, layer_idx: int) -> bool:
    """Whether ``layer_idx`` is a full/global-attention layer."""
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        return layer_types[layer_idx] == "full_attention"
    # Fallback: 5:1 sliding:full -> every 6th layer is full.
    pattern = getattr(config, "sliding_window_pattern", 6)
    return (layer_idx + 1) % pattern == 0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """transformers ``rotate_half`` (NEOX/half-split over the last dim)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _GemmaPartialRoPE(nn.Module):
    """Gemma-4 'proportional' partial rotary for full/global layers.

    transformers rotates the first ``rope_angles`` frequencies over the FULL
    head_dim half-split — pairing dim ``i`` with ``i + head_dim/2`` and leaving
    the remaining dims unrotated (``inv_freq`` zero-padded). This is NOT the same
    as Apple's composite :class:`RoPE` with ``dims < head_dim``, which pairs
    ``i`` with ``i + dims/2`` inside a contiguous ``dims``-wide block. The two
    layouts differ (verified: ~21 dB vs the reference on the global layer), so we
    reproduce ``transformers`` ``_compute_proportional_rope_parameters`` +
    ``apply_rotary_pos_emb`` exactly. Frequencies are identical to the composite;
    only the dim pairing is corrected.
    """

    def __init__(self, head_dim: int, rope_angles: int, base: float) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.rope_angles = int(rope_angles)
        self.base = float(base)

    def forward(
        self,
        input: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        freqs: torch.Tensor | None = None,
        offset: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        # Recompute inv_freq in float32 every call — deliberately NOT a registered buffer. A buffer
        # is cast to the model's compute dtype by ``model.to(dtype=bf16)``, and bf16's ~3-digit
        # mantissa corrupts these small frequencies (1/theta^(i/256)) → wrong full-layer rotation →
        # incoherent output. Recomputing in f32 (traced once for export) keeps full precision.
        # input: [B, n_heads, S, head_dim]; position_ids: [B, S].
        half = self.head_dim // 2
        dev = input.device
        idx = torch.arange(0, 2 * self.rope_angles, 2, dtype=torch.float32, device=dev)
        active = 1.0 / (self.base ** (idx / self.head_dim))
        inv_freq = torch.cat(
            [active, torch.zeros(half - self.rope_angles, dtype=torch.float32, device=dev)], dim=0)
        pos = position_ids.float()
        ang = pos[..., None] * inv_freq                              # [B, S, half]
        emb = torch.cat((ang, ang), dim=-1)                          # [B, S, head_dim]
        cos = emb.cos().unsqueeze(1).to(input.dtype)                 # [B, 1, S, head_dim]
        sin = emb.sin().unsqueeze(1).to(input.dtype)
        return input * cos + _rotate_half(input) * sin


def _rope_for_layer(config, layer_idx: int, head_dim: int):
    """Build the per-layer RoPE.

    Full/global layers: theta 1e6, partial rotary (``partial_rotary_factor``,
    default 0.25) via :class:`_GemmaPartialRoPE`. Sliding layers: theta 1e4, full
    rotary via the composite :class:`RoPE`.
    """
    is_global = _is_global_layer(config, layer_idx)
    params = getattr(config, "rope_parameters", None) or {}
    key = "full_attention" if is_global else "sliding_attention"
    entry = params.get(key) if isinstance(params, dict) else None
    if isinstance(entry, dict):
        base = float(entry.get("rope_theta", 1_000_000.0 if is_global else 10_000.0))
        prf = entry.get("partial_rotary_factor")
    else:
        base = 1_000_000.0 if is_global else 10_000.0
        prf = 0.25 if is_global else None

    if prf is not None and float(prf) < 1.0:
        # Gemma-4 'proportional' partial rotary. The composite RoPE's `dims`-partial
        # mode pairs i<->i+dims/2 in a contiguous block, but Gemma needs the full
        # head_dim half-split (pair i<->i+head_dim/2, only the first `rope_angles`
        # freqs active). Use a faithful reimplementation instead.
        rope_angles = int(float(prf) * head_dim // 2)
        return _GemmaPartialRoPE(head_dim, rope_angles, base)
    return RoPE(base=base, dims=None)


def _pack_kv(t: torch.Tensor, cache_kv_heads: int, cache_head_dim: int) -> torch.Tensor:
    """Pack native K/V ``[B, n_kv, S, hd]`` into the uniform cache layout
    ``[B, cache_kv_heads, S, cache_head_dim]`` (flatten channels, zero-pad)."""
    b, n_kv, s, hd = t.shape
    native = n_kv * hd
    cache_w = cache_kv_heads * cache_head_dim
    flat = t.permute(0, 2, 1, 3).reshape(b, s, native)
    if native < cache_w:
        flat = F.pad(flat, (0, cache_w - native))
    return flat.reshape(b, s, cache_kv_heads, cache_head_dim).permute(0, 2, 1, 3)


def _unpack_kv(t: torch.Tensor, n_kv: int, hd: int) -> torch.Tensor:
    """Inverse of :func:`_pack_kv`: ``[B, cache_kv_heads, S, cache_head_dim]``
    -> native ``[B, n_kv, S, hd]`` (slice off the zero padding)."""
    b, ckv, s, chd = t.shape
    flat = t.permute(0, 2, 1, 3).reshape(b, s, ckv * chd)
    flat = flat.narrow(-1, 0, n_kv * hd)
    return flat.reshape(b, s, n_kv, hd).permute(0, 2, 1, 3)


def _rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Weightless RMS normalization over the last dim.

    Gemma-4 applies ``v_norm`` = ``Gemma4RMSNorm(head_dim, with_scale=False)`` to
    the value states (no learned weight). Computed in float and cast back.
    """
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.to(dtype)


class MLP(nn.Module):
    """Gated MLP with gelu(tanh) activation (Gemma)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_tensor = self.up_proj(x)
        gate_tensor = nn.functional.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(up_tensor * gate_tensor)


class GeluGLU(nn.Module):
    """Gemma gated activation for ``SwitchGLU``: ``gelu_tanh(gate) * up``."""

    def forward(self, up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate, approximate="tanh") * up


class Gemma4SparseMoE(nn.Module):
    """Gemma-4 128-expert top-8 MoE (the ``A4B`` sparse branch). Grounded in the official
    ``transformers`` ``Gemma4TextRouter`` / ``Gemma4TextExperts``:

        router: logits = proj( rmsnorm_noscale(x) * scale * (1/sqrt(hidden)) )
                probs = softmax(logits); top-k; weights = probs[topk] / sum; weights *= per_expert_scale
        experts: gelu_tanh(gate) * up -> down, weighted by the router weights and summed.

    The per-expert ``router.per_expert_scale`` is folded into the expert ``down_proj`` at load
    (mathematically identical to scaling the router weights), so it isn't applied here. Decode is
    seq=1, so ``SwitchGLU``'s per-token top-k gather is the efficient path (no dense rewrite).
    """

    def __init__(self, config) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.eps = config.rms_norm_eps
        self._inv_sqrt_hidden = hidden**-0.5
        self.router_proj = nn.Linear(hidden, self.num_experts, bias=False)
        self.router_scale = nn.Parameter(torch.zeros(hidden))
        self.experts = SwitchGLU(
            hidden, config.moe_intermediate_size, self.num_experts, bias=False,
            activation=GeluGLU())

    def _router_norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x_experts: torch.Tensor, x_router: torch.Tensor) -> torch.Tensor:
        tmp = self._router_norm(x_router) * self._inv_sqrt_hidden * self.router_scale
        logits = self.router_proj(tmp)
        gates = torch.softmax(logits, dim=-1, dtype=torch.float32)
        scores, indices = torch.topk(gates, self.top_k, dim=-1, largest=True)
        scores = scores / torch.sum(scores, dim=-1, keepdim=True)
        indices = indices.to(torch.uint16)
        y = self.experts(x_experts, indices)  # [b, s, top_k, hidden]
        y = y * scores.unsqueeze(-1).to(y.dtype)
        return torch.sum(y, dim=-2).to(x_experts.dtype)


class Embedding(nn.Embedding):
    """Embedding scaled by ``sqrt(hidden_size)`` (Gemma normalizer)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None,
        embed_scale: float = 1.0,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * torch.tensor(
            self.embed_scale, dtype=self.weight.dtype, device=self.weight.device
        )


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #


class Gemma4Attention(nn.Module):
    """Gemma-4 self-attention (full text decoder).

    Sliding layers have separate ``q/k/v_proj`` + ``q/k_norm``. Full/global
    layers (``attention_k_eq_v``) share one projection for K and V, so there is
    no ``v_proj`` weight and V is the un-normed K projection.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.is_global = is_global = _is_global_layer(config, layer_idx)
        self.k_eq_v = bool(getattr(config, "attention_k_eq_v", False)) and is_global

        self.n_heads = n_heads = config.num_attention_heads
        if is_global:
            # getattr(..., default) returns the default ONLY when the attribute is absent; the elastic
            # E-models (E2B/E4B) set num_global_key_value_heads (and can set global_head_dim) to an
            # explicit None, so fall back with `or`. Verified against the E2B checkpoint: global layers
            # use num_key_value_heads kv-heads at global_head_dim (e.g. 1 head x 512).
            self.head_dim = head_dim = getattr(config, "global_head_dim", None) or config.head_dim
            self.n_kv_heads = n_kv_heads = (
                getattr(config, "num_global_key_value_heads", None) or config.num_key_value_heads
            )
        else:
            self.head_dim = head_dim = config.head_dim
            self.n_kv_heads = n_kv_heads = config.num_key_value_heads

        # Uniform export cache geometry (shared across all layers). Must hold the WIDEST layer: most
        # Gemma-4 sizes have local width (num_key_value_heads*head_dim) >= global, but the elastic
        # E-models use global_head_dim (512) with the same kv-head count, so their global layers
        # (1 kv x 512) are WIDER than local (1 kv x 256) -> _pack_kv's reshape would overflow. Size to
        # the max width. For non-E models local is the max, so cache_head_dim stays head_dim (unchanged).
        _gkv = getattr(config, "num_global_key_value_heads", None) or config.num_key_value_heads
        _ghd = getattr(config, "global_head_dim", None) or config.head_dim
        _max_w = max(config.num_key_value_heads * config.head_dim, _gkv * _ghd)
        self.cache_kv_heads = config.num_key_value_heads
        self.cache_head_dim = _max_w // config.num_key_value_heads

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        if not self.k_eq_v:
            self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rms_norm_eps = eps = config.rms_norm_eps
        self.q_norm = Gemma4RMSNorm(head_dim, eps=eps)
        self.k_norm = Gemma4RMSNorm(head_dim, eps=eps)

        self.rope = _rope_for_layer(config, layer_idx, head_dim)
        # Gemma-4 attention uses scale=1.0 (q_norm/k_norm control magnitudes),
        # NOT 1/sqrt(head_dim).
        self.sdpa = SDPA(
            scale=1.0,
            window_size=config.sliding_window if not is_global else 0,
            is_causal=True,
        )

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, _ = x.shape
        q = self.q_proj(x).reshape(b, s, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k_raw = self.k_proj(x).reshape(b, s, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        # V is the un-normed projection (v_proj, or k_proj when k_eq_v), passed
        # through the weightless v_norm. K is separately k_norm'd. V is not RoPE'd.
        if self.k_eq_v:
            v = k_raw
        else:
            v = self.v_proj(x).reshape(b, s, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_norm(k_raw)
        v = _rms_normalize(v, self.rms_norm_eps)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
        shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_repr_kv: bool = False,
    ) -> torch.Tensor:
        # ``shared_kv`` is the EAGLE draft's cross-attention KV (consumed by
        # ``Gemma4AssistantAttention``); the full text decoder ignores it.
        batch_size, query_len, _ = x.shape
        q, k, v = self._project(x)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        q = self.rope(q, position_ids=rope_positions)
        k = self.rope(k, position_ids=rope_positions)

        # EAGLE: the draft cross-attends to the target's *new-position* repr K/V
        # at this layer (post k_norm+RoPE for K, post v_norm for V), shape
        # ``[B, n_kv, query_len, head_dim]``. Capture BEFORE the cache merge so the
        # value is a normal intermediate tensor — NOT a read of the persistent KV
        # state (reading mutated state as a graph output breaks the in-place
        # ``mutable_slice_update`` under torch.export functionalization). Swift
        # accumulates these new positions into the full prefix host-side.
        repr_k, repr_v = (k, v) if return_repr_kv else (None, None)

        if cache is not None:
            k_packed = _pack_kv(k, self.cache_kv_heads, self.cache_head_dim)
            v_packed = _pack_kv(v, self.cache_kv_heads, self.cache_head_dim)
            k_packed, v_packed = cache.update_and_fetch(
                self.layer_idx, offset, k_packed, v_packed, seq_len=seq_len, query_len=query_len
            )
            k = _unpack_kv(k_packed, self.n_kv_heads, self.head_dim)
            v = _unpack_kv(v_packed, self.n_kv_heads, self.head_dim)

        output = (
            self.sdpa(query=q, key=k, value=v)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, self.n_heads * self.head_dim)
        )
        out = self.o_proj(output)
        if return_repr_kv:
            return out, repr_k, repr_v
        return out


class Gemma4AssistantAttention(Gemma4Attention):
    """EAGLE draft cross-attention: only ``q_proj`` / ``q_norm`` / ``o_proj`` exist.

    The draft has no ``k_proj`` / ``v_proj`` (``num_kv_shared_layers`` == all
    layers). Each layer's locally-projected, RMS-normed, RoPE'd query attends to
    the TARGET model's precomputed K/V for this layer's type (``shared_kv``):
    sliding layers consume the backbone's last sliding-layer K/V, the full layer
    consumes the last full-layer K/V. Those K/V already carry the backbone's
    ``k_norm`` + RoPE (K) and ``v_norm`` (V), so the draft applies none of that to
    them. This is the faithful ``transformers`` ``Gemma4AssistantForCausalLM``
    mechanism (cross-attention from a single, constant query position).
    """

    def __init__(self, config, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.is_global = is_global = _is_global_layer(config, layer_idx)
        self.k_eq_v = True

        self.n_heads = n_heads = config.num_attention_heads
        if is_global:
            self.head_dim = head_dim = getattr(config, "global_head_dim", config.head_dim)
            self.n_kv_heads = getattr(
                config, "num_global_key_value_heads", config.num_key_value_heads
            )
        else:
            self.head_dim = head_dim = config.head_dim
            self.n_kv_heads = config.num_key_value_heads

        self.cache_kv_heads = config.num_key_value_heads
        self.cache_head_dim = config.head_dim

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.rms_norm_eps = config.rms_norm_eps
        self.q_norm = Gemma4RMSNorm(head_dim, eps=config.rms_norm_eps)

        self.rope = _rope_for_layer(config, layer_idx, head_dim)
        self.layer_type = "full_attention" if is_global else "sliding_attention"
        # Cross-attention from a single, constant query position over the full
        # supplied KV: no causal mask (the one query sees all keys). Sliding-window
        # cropping, when a context exceeds the window, is applied host-side to the
        # K/V tensors fed in (so the graph stays window-agnostic).
        self.sdpa = SDPA(scale=1.0, window_size=0, is_causal=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
        shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).reshape(b, s, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        q = self.rope(q, position_ids=position_ids)
        k, v = shared_kv[self.layer_type]
        output = (
            self.sdpa(query=q, key=k, value=v)
            .permute(0, 2, 1, 3)
            .reshape(b, s, self.n_heads * self.head_dim)
        )
        return self.o_proj(output)


# --------------------------------------------------------------------------- #
# Decoder block / model
# --------------------------------------------------------------------------- #


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int, attention_cls=Gemma4Attention) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = attention_cls(config, layer_idx)
        self.mlp = MLP(hidden_size, config.intermediate_size)

        eps = config.rms_norm_eps
        self.input_layernorm = Gemma4RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = Gemma4RMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(hidden_size, eps=eps)
        # MoE (``A4B``) branch: a dense shared expert (``mlp``) PLUS a 128-expert top-8 MoE, each
        # with its own pre/post feed-forward norm, summed before ``post_feedforward_layernorm``.
        # Matches the official ``Gemma4TextDecoderLayer`` (enable_moe_block). Off for the dense 31B.
        self.enable_moe = (
            getattr(config, "enable_moe_block", False) and getattr(config, "num_experts", 0) > 0)
        if self.enable_moe:
            self.moe = Gemma4SparseMoE(config)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(hidden_size, eps=eps)
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(hidden_size, eps=eps)
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(hidden_size, eps=eps)
        # Per-layer learned scalar that multiplies the whole layer output (applied
        # at the very end, after both residual additions).
        self.layer_scalar = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
        shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_repr_kv: bool = False,
    ) -> torch.Tensor:
        if return_repr_kv:
            r, repr_k, repr_v = self.self_attn(
                self.input_layernorm(x), position_ids, cache, shared_kv, return_repr_kv=True
            )
        else:
            r = self.self_attn(self.input_layernorm(x), position_ids, cache, shared_kv)
        h = x + self.post_attention_layernorm(r)  # attn residual (the FF input)
        mlp_out = self.mlp(self.pre_feedforward_layernorm(h))
        if self.enable_moe:
            # Dual feed-forward: shared MLP + 128-expert MoE (router on the raw attn residual,
            # experts on a separately-normed copy), each post-normed, summed. Matches the official
            # Gemma4TextDecoderLayer MoE block exactly.
            mlp_out = self.post_feedforward_layernorm_1(mlp_out)
            moe_out = self.post_feedforward_layernorm_2(
                self.moe(self.pre_feedforward_layernorm_2(h), h))
            r = self.post_feedforward_layernorm(mlp_out + moe_out)
        else:
            r = self.post_feedforward_layernorm(mlp_out)
        h = h + r
        out = h * self.layer_scalar
        if return_repr_kv:
            return out, repr_k, repr_v
        return out


class Gemma4Model(nn.Module):
    def __init__(self, config, attention_cls=Gemma4Attention) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.embed_tokens = Embedding(
            config.vocab_size,
            hidden_size,
            getattr(config, "pad_token_id", None),
            embed_scale=hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, attention_cls)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma4RMSNorm(hidden_size, eps=config.rms_norm_eps)
        # EAGLE: when set to ``(full_layer_idx, sliding_layer_idx)`` the forward
        # also returns the new-position repr K/V from those two layers (keyed by
        # ``layer_type``) for the draft to cross-attend to. ``None`` => standard
        # logits-only behaviour (returns just the hidden state).
        self.eagle_repr_layers: tuple[int, int] | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        repr_layers = self.eagle_repr_layers
        if repr_layers is None:
            for layer in self.layers:
                h = layer(h, position_ids, cache)
            return self.norm(h)
        # EAGLE repr-capture path: collect the two repr layers' new-position K/V.
        full_idx, slide_idx = repr_layers
        repr_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == full_idx:
                h, k_new, v_new = layer(h, position_ids, cache, return_repr_kv=True)
                repr_kv["full_attention"] = (k_new, v_new)
            elif layer_idx == slide_idx:
                h, k_new, v_new = layer(h, position_ids, cache, return_repr_kv=True)
                repr_kv["sliding_attention"] = (k_new, v_new)
            else:
                h = layer(h, position_ids, cache)
        return self.norm(h), repr_kv


# --------------------------------------------------------------------------- #
# ForCausalLM
# --------------------------------------------------------------------------- #


class Gemma4ForCausalLM(BaseForCausalLM):
    """Gemma-4 text decoder (``gemma4``).

    Loads from the multimodal ``Gemma4ForConditionalGeneration`` checkpoint:
    text weights live under ``model.language_model.`` (registry strips the
    leading ``model.`` prefix; :meth:`load_state_dict` maps the residual
    ``language_model.`` segment onto ``model.`` and drops vision/audio keys).
    """

    _HF_MODEL_CLASS = None  # transformers has no Gemma-4 class; load via safetensors.
    _ATTENTION_CLS = Gemma4Attention

    @override
    def _init_model(self, config) -> None:
        self.model = Gemma4Model(config, self._ATTENTION_CLS)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.final_logit_softcapping = getattr(config, "final_logit_softcapping", None)

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
        logits = self.lm_head(out)
        if self.final_logit_softcapping is not None:
            cap = float(self.final_logit_softcapping)
            logits = torch.tanh(logits / cap) * cap
        return logits

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Projections are kept separate (no qkv fusion): the global layers omit
        # v_proj (k_eq_v) and use a different head_dim, which makes fusion
        # ill-defined. Drop non-text (vision/audio) keys if present so a full
        # (non-streaming) load doesn't choke on them.
        for key in list(state_dict.keys()):
            if key.startswith(("vision_tower", "embed_vision", "audio")) or (
                ".vision_tower." in key or ".embed_vision." in key
            ):
                del state_dict[key]
        # Gemma-4 MoE (``A4B``): remap the HF sparse-expert weights onto the ``SwitchGLU`` layout
        # ``Gemma4SparseMoE`` expects. Split the fused ``experts.gate_up_proj`` into gate/up, stack
        # experts into ``[1, E, out, in]``, fold ``router.per_expert_scale`` into the expert
        # ``down_proj`` (exact — it scales each expert's output), and rename the router. Grounded in
        # the official ``Gemma4TextRouter`` / ``Gemma4TextExperts``. No-op for the dense 31B.
        import re as _re

        moe_layers = sorted(
            {
                int(m.group(1))
                for k in state_dict
                if (m := _re.search(r"\.layers\.(\d+)\.experts\.gate_up_proj$", k))
            }
        )
        for li in moe_layers:
            p = f"model.layers.{li}."
            gate_up = state_dict.pop(p + "experts.gate_up_proj")  # [E, 2*ff, hidden]
            n_ff = gate_up.shape[1] // 2
            state_dict[p + "moe.experts.gate_proj.weight"] = (
                gate_up[:, :n_ff, :].unsqueeze(0).contiguous())
            state_dict[p + "moe.experts.up_proj.weight"] = (
                gate_up[:, n_ff:, :].unsqueeze(0).contiguous())
            down = state_dict.pop(p + "experts.down_proj")  # [E, hidden, ff]
            pes = state_dict.pop(p + "router.per_expert_scale").to(down.dtype)  # [E]
            state_dict[p + "moe.experts.down_proj.weight"] = (
                (down * pes.view(-1, 1, 1)).unsqueeze(0).contiguous())
            state_dict[p + "moe.router_proj.weight"] = state_dict.pop(p + "router.proj.weight")
            state_dict[p + "moe.router_scale"] = state_dict.pop(p + "router.scale")

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # HF text keys arrive as `language_model.<...>` after the registry strips
        # the leading `model.` prefix. Map them onto this model's `model.<...>`
        # namespace and drop any remaining vision/audio keys.
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(("vision_tower.", "embed_vision.", "audio")):
                continue
            if key.startswith("language_model."):
                key = "model." + key[len("language_model.") :]
            remapped[key] = value
        super().load_state_dict(remapped, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def from_hf_memory_efficient(  # noqa: PLR0913
        cls,
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        hf_config_attr: str | None = None,
        hf_state_dict_prefix: str = "",
    ):
        """Stream the Gemma-4 text decoder out of the multimodal checkpoint.

        The base loader keys per-layer streaming off a hardcoded
        ``model.layers.N`` pattern, but Gemma-4's text weights live under
        ``model.language_model.layers.N`` (vision/audio under sibling prefixes).
        With the base pattern every tensor falls into the one-shot "shared"
        bucket, loading the whole 31B model into RAM at once (OOM). This
        override indexes per-layer correctly, remaps keys onto this model's
        ``model.*`` namespace, and skips non-text modalities, so peak RAM stays
        ~one transformer layer.
        """
        import gc
        import os
        import re

        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoConfig

        from coreai_models.models import base as _b
        from coreai_models.primitives.macos.cache import KVCache as _KV

        text_prefix = hf_state_dict_prefix or "model.language_model."

        model_dir = snapshot_download(
            huggingface_model_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )
        raw_config = AutoConfig.from_pretrained(model_dir)
        hf_config = getattr(raw_config, hf_config_attr) if hf_config_attr else raw_config
        config = cls._get_reauthored_config(hf_config, max_context_length, num_layers=num_layers)
        model = cls(config, model_device="meta")
        model.to(dtype=target_dtype)

        def remap(k: str) -> str:
            k = k[len(text_prefix) :] if k.startswith(text_prefix) else k
            return k if k.startswith("model.") else "model." + k

        files = _b._resolve_safetensors_files(model_dir)
        layer_re = re.compile(re.escape(text_prefix) + r"layers\.(\d+)\.")
        per_layer: dict[int, dict[str, str]] = {}
        shared: dict[str, str] = {}
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if not key.startswith(text_prefix):
                        continue  # skip vision / audio / projector weights
                    m = layer_re.match(key)
                    if m:
                        li = int(m.group(1))
                        if num_layers is not None and li >= num_layers:
                            continue
                        per_layer.setdefault(li, {})[key] = path
                    else:
                        shared[key] = path

        # Shared params (embed_tokens, norm) first.
        shared_t = {
            remap(k): v for k, v in _b._load_tensors_for_keys(shared, target_dtype).items()
        }
        if mmap_path is not None:
            os.makedirs(mmap_path, exist_ok=True)
            _b._save_and_mmap_safetensors(
                model, shared_t, os.path.join(mmap_path, "shared.safetensors")
            )
        else:
            model.load_state_dict(shared_t, assign=True, strict=False)
        del shared_t
        gc.collect()

        # One transformer layer at a time.
        exclude = {_KV.HF_K_BUFFER_NAME, _KV.HF_V_BUFFER_NAME}
        for li in sorted(per_layer):
            sd = {
                remap(k): v for k, v in _b._load_tensors_for_keys(per_layer[li], target_dtype).items()
            }
            model._mutate_state_dict(sd)
            if mmap_path is not None:
                lp = f"model.layers.{li}."
                rel = {k[len(lp) :]: v for k, v in sd.items() if k[len(lp) :] not in exclude}
                _b._save_and_mmap_safetensors(
                    model.model.layers[li], rel, os.path.join(mmap_path, f"layer_{li}.safetensors")
                )
            else:
                model.load_state_dict(sd, assign=True, strict=False)
            del sd
            gc.collect()

        if config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        meta = [n for n, p in model.named_parameters() if p.is_meta]
        if meta:
            raise RuntimeError(f"Parameters not loaded: {meta}")
        return model


class Gemma4AssistantForCausalLM(Gemma4ForCausalLM):
    """Faithful EAGLE/MTP ``gemma4_assistant`` draft (``SinglePositionMultiToken``).

    Reimplements Google's ``transformers`` ``Gemma4AssistantForCausalLM`` for the
    Core AI export path. Per draft micro-step the host supplies the last seen
    token, the target's last hidden state, a constant position, and the target's
    representative K/V (``shared_kv``: last full-layer + last sliding-layer K/V).
    The draft:

    1. embeds the token with the *target's* scaled embedding (``target_embed``),
       concatenates ``[embed, hidden]`` (backbone-width each) and projects down
       via ``pre_projection`` (2*backbone -> draft hidden);
    2. runs 4 decoder layers whose queries cross-attend to ``shared_kv`` (no draft
       k/v; see :class:`Gemma4AssistantAttention`), at the constant position;
    3. emits token ``logits`` (tied draft head) AND a backbone-width next-hidden
       (``post_projection``) for its own next micro-step.

    ``target_embed`` is not in the assistant checkpoint — it is injected from the
    TARGET checkpoint at convert time (see :meth:`set_target_embed`).
    """

    _HF_MODEL_CLASS = None
    _ATTENTION_CLS = Gemma4AssistantAttention

    @override
    def _init_model(self, config) -> None:
        super()._init_model(config)  # model + tied lm_head + final_logit_softcapping(None)
        backbone = getattr(config, "backbone_hidden_size", None)
        if not backbone:
            raise ValueError(
                "Gemma4AssistantForCausalLM needs config.backbone_hidden_size (the "
                "target's hidden_size) for the EAGLE pre/post projections.")
        self.backbone_hidden_size = int(backbone)
        # EAGLE cross-model glue (KEPT — this is the whole point of the draft).
        self.pre_projection = nn.Linear(2 * backbone, config.hidden_size, bias=False)
        self.post_projection = nn.Linear(config.hidden_size, backbone, bias=False)
        # The TARGET's scaled input embedding (backbone-width), used to embed the
        # last seen token for the pre_projection input. Injected from the target.
        self.target_embed = Embedding(
            config.vocab_size, backbone, getattr(config, "pad_token_id", None),
            embed_scale=backbone**0.5)

    def set_target_embed(self, weight: torch.Tensor) -> None:
        """Install the TARGET model's ``embed_tokens.weight`` (``[vocab, backbone]``)."""
        with torch.no_grad():
            self.target_embed.weight = nn.Parameter(
                weight.to(self.target_embed.weight.dtype), requires_grad=False)

    def forward(  # type: ignore[override]
        self,
        token_id: torch.Tensor,
        hidden: torch.Tensor,
        position_ids: torch.IntTensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        k_sliding: torch.Tensor,
        v_sliding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # inputs_embeds = [ target_embed(token) ; target_hidden ]  (embed first)
        emb = self.target_embed(token_id)                       # [B, 1, backbone]
        h = self.pre_projection(torch.cat([emb, hidden], dim=-1))  # [B, 1, draft_hidden]
        shared_kv = {
            "full_attention": (k_full, v_full),
            "sliding_attention": (k_sliding, v_sliding),
        }
        for layer in self.model.layers:
            h = layer(h, position_ids, None, shared_kv)
        h = self.model.norm(h)                                  # [B, 1, draft_hidden]
        next_hidden = self.post_projection(h)                   # [B, 1, backbone]
        logits = self.lm_head(h)                                # [B, 1, vocab]
        return logits, next_hidden

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Assistant keys map 1:1 onto this module (model.*, lm_head, pre/post
        # projection). target_embed is injected separately. No remap needed.
        return

    def draft_unrolled(
        self,
        token_id: torch.Tensor,
        hidden: torch.Tensor,
        position_ids: torch.IntTensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        k_sliding: torch.Tensor,
        v_sliding: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Run ``num_steps`` draft micro-steps in ONE pass (in-graph argmax + recurrence),
        returning the proposed draft token ids ``[B, num_steps]`` (int32). The shared K/V and
        position are CONSTANT across steps (the SinglePosition MTP mechanism); only the token and
        the predicted hidden recur. Unrolled at export so K becomes one Core AI dispatch."""
        tok = token_id
        h = hidden
        out_tokens: list[torch.Tensor] = []
        for _ in range(int(num_steps)):
            logits, h = self.forward(tok, h, position_ids, k_full, v_full, k_sliding, v_sliding)
            tok = torch.argmax(logits[:, -1:, :], dim=-1).to(torch.int32)  # [B, 1]
            out_tokens.append(tok)
        return torch.cat(out_tokens, dim=-1)  # [B, num_steps]


class Gemma4AssistantUnrolled(nn.Module):
    """Export wrapper that unrolls ``num_steps`` EAGLE draft micro-steps into a single Core AI
    graph. Inputs match the single-step draft (token_id, hidden, position_ids, k_full, v_full,
    k_sliding, v_sliding); output is the ``[1, num_steps]`` int32 proposed draft tokens. One
    dispatch replaces the K sequential draft calls — removing the per-step GPU launch tax that is
    the dominant cost once acceptance is high."""

    def __init__(self, draft: "Gemma4AssistantForCausalLM", num_steps: int) -> None:
        super().__init__()
        self.draft = draft
        self.num_steps = int(num_steps)

    def forward(
        self,
        token_id: torch.Tensor,
        hidden: torch.Tensor,
        position_ids: torch.IntTensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        k_sliding: torch.Tensor,
        v_sliding: torch.Tensor,
    ) -> torch.Tensor:
        return self.draft.draft_unrolled(
            token_id, hidden, position_ids, k_full, v_full, k_sliding, v_sliding, self.num_steps)

    def load_state_dict(self, state_dict, strict: bool = False, assign: bool = False):
        # ``target_embed.*`` is never in the assistant checkpoint (it comes from
        # the target), so load non-strict and re-tie the draft head.
        BaseForCausalLM.load_state_dict(self, state_dict, strict=False, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def from_hf_memory_efficient(cls, *args, **kwargs):
        # The assistant checkpoint has no multimodal wrapper — its keys are
        # already `model.layers.N`, so the base streaming loader works directly
        # (skip Gemma4ForCausalLM's `model.language_model.` override).
        return BaseForCausalLM.from_hf_memory_efficient.__func__(cls, *args, **kwargs)


def _eagle_repr_layer_indices(config) -> tuple[int, int]:
    """(last full-attention layer, last sliding-attention layer) — the layers whose
    K/V the EAGLE draft cross-attends to (``shared_kv_states`` per ``layer_type``)."""
    n = config.num_hidden_layers
    lt = [
        "full_attention" if _is_global_layer(config, i) else "sliding_attention"
        for i in range(n)
    ]
    last_full = max(i for i in range(n) if lt[i] == "full_attention")
    last_slide = max(i for i in range(n) if lt[i] == "sliding_attention")
    return last_full, last_slide


class Gemma4EagleTarget(nn.Module):
    """Export wrapper that turns a :class:`Gemma4ForCausalLM` target into the EAGLE
    target contract: emit ``(logits, hidden, k_full, v_full, k_sliding, v_sliding)``.

    ``hidden`` is the final post-norm hidden state (== ``transformers``
    ``hidden_states[-1]``). The four KV tensors are the representative
    full/sliding-layer K/V (``shared_kv_states``) the draft attends to, read back
    from the (post-``k_norm``+RoPE / post-``v_norm``) packed KV cache after the
    forward and unpacked to native ``[B, n_kv, seq, head_dim]``. Drives the draft
    via :class:`Gemma4AssistantForCausalLM`.
    """

    def __init__(self, base: Gemma4ForCausalLM) -> None:
        super().__init__()
        self.base = base
        cfg = base.config
        self.last_full, self.last_slide = _eagle_repr_layer_indices(cfg)
        # Drive the model's repr-capture path: it returns the new-position K/V at
        # these two layers as normal graph outputs (no persistent-state read).
        self.base.model.eagle_repr_layers = (self.last_full, self.last_slide)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cache = KVCache(k_cache, v_cache)
        # Model returns (post-norm hidden, repr_kv) — repr_kv holds the *new
        # positions'* K/V for this forward at the full + sliding repr layers.
        hidden, repr_kv = self.base.model(input_ids, position_ids, cache)
        logits = self.base.lm_head(hidden)
        if self.base.final_logit_softcapping is not None:
            cap = float(self.base.final_logit_softcapping)
            logits = torch.tanh(logits / cap) * cap
        k_full, v_full = repr_kv["full_attention"]
        k_sliding, v_sliding = repr_kv["sliding_attention"]
        # Cast outputs to f16: bf16-compute graphs read back all-zero through
        # ``program.optimize()`` unless the output is f16 (the same workaround the
        # base logits use via ``cast_logits_bfloat16_to_float16``). The EAGLE draft
        # consumes f16 hidden/KV, so this is also the consumer's expected dtype.
        # NOTE: these are the NEW-position K/V only ([B, n_kv, query_len, hd]);
        # the Swift target engine accumulates them into the full prefix host-side
        # (and applies sliding-window cropping before feeding the draft).
        f16 = torch.float16
        return (
            logits.to(f16), hidden.to(f16),
            k_full.to(f16), v_full.to(f16), k_sliding.to(f16), v_sliding.to(f16),
        )
