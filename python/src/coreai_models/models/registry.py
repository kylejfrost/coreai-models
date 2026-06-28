# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Model registry mapping HuggingFace model_type to model classes."""

from dataclasses import dataclass
from functools import lru_cache

import torch.nn as nn

# Gemma-4 is not in transformers yet. Import its module eagerly (here, at
# registry import time, which the export CLI triggers before
# `AutoConfig.from_pretrained` is called) so its `AutoConfig` /
# `AutoTokenizer` registration runs in time.
from coreai_models.models.macos import gemma4 as _gemma4  # noqa: F401

# Qwen3.5 is likewise not in transformers yet; import eagerly so its AutoConfig
# registration runs before the export CLI calls `AutoConfig.from_pretrained`.
from coreai_models.models.macos import qwen3_5 as _qwen3_5  # noqa: F401
from coreai_models.models.macos import qwen3_5_moe as _qwen3_5_moe  # noqa: F401


@dataclass
class ModelEntry:
    """Registry entry for a model family."""

    macos_class: type[nn.Module] | None = None
    ios_class: type[nn.Module] | None = None
    # Multimodal-checkpoint hooks consumed by `from_hf_memory_efficient` and the
    # `text_config` unwrap in the export pipeline.
    # `hf_config_attr`: attribute on the top-level HF config holding the
    #     per-modality sub-config (e.g. "text_config" for Gemma-3).
    # `hf_state_dict_prefix`: prefix on safetensors keys for this modality
    #     (e.g. "language_model." for Gemma-3). Stripped before assignment.
    hf_config_attr: str | None = None
    hf_state_dict_prefix: str = ""


@lru_cache(maxsize=1)
def _get_registry() -> dict[str, ModelEntry]:
    """Build the model registry (cached singleton). Lazy imports to avoid circular deps."""
    from coreai_models.models.ios.mistral import MistralForCausalLMForiOS
    from coreai_models.models.ios.qwen2 import Qwen2ForCausalLMForiOS
    from coreai_models.models.ios.qwen3 import Qwen3ForCausalLMForiOS
    from coreai_models.models.macos.gemma3_text import Gemma3ForCausalLM
    from coreai_models.models.macos.gemma4 import (
        Gemma4AssistantForCausalLM,
        Gemma4ForCausalLM,
    )
    from coreai_models.models.macos.glm4 import Glm4ForCausalLM
    from coreai_models.models.macos.gpt_oss import GptOssForCausalLM
    from coreai_models.models.macos.mistral import MistralForCausalLM
    from coreai_models.models.macos.mixtral import MixtralForCausalLM
    from coreai_models.models.macos.qwen2 import Qwen2ForCausalLM
    from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM
    from coreai_models.models.macos.qwen3_5 import Qwen3_5ForCausalLM
    from coreai_models.models.macos.qwen3_5_moe import Qwen3_5MoeForCausalLM
    from coreai_models.models.macos.qwen3_moe import Qwen3MoeForCausalLM

    return {
        "gemma3_text": ModelEntry(
            macos_class=Gemma3ForCausalLM,
            hf_config_attr="text_config",
            hf_state_dict_prefix="language_model.",
        ),
        "gemma4": ModelEntry(
            macos_class=Gemma4ForCausalLM,
            hf_config_attr="text_config",
            hf_state_dict_prefix="model.language_model.",
        ),
        "gemma4_assistant": ModelEntry(
            macos_class=Gemma4AssistantForCausalLM,
            hf_config_attr="text_config",
        ),
        "glm4": ModelEntry(
            macos_class=Glm4ForCausalLM,
        ),
        "gpt_oss": ModelEntry(
            macos_class=GptOssForCausalLM,
        ),
        "mistral": ModelEntry(
            macos_class=MistralForCausalLM,
            ios_class=MistralForCausalLMForiOS,
        ),
        "mixtral": ModelEntry(
            macos_class=MixtralForCausalLM,
        ),
        "qwen2": ModelEntry(
            macos_class=Qwen2ForCausalLM,
            ios_class=Qwen2ForCausalLMForiOS,
        ),
        "qwen3": ModelEntry(
            macos_class=Qwen3ForCausalLM,
            ios_class=Qwen3ForCausalLMForiOS,
        ),
        "qwen3_moe": ModelEntry(
            macos_class=Qwen3MoeForCausalLM,
        ),
        "qwen3_5": ModelEntry(
            macos_class=Qwen3_5ForCausalLM,
            hf_config_attr="text_config",
            # Empty prefix: text weights live under `model.language_model.` but
            # the untied `lm_head.weight` is top-level (outside `model.`), so we
            # take the whole checkpoint and remap/strip in `load_state_dict`.
            hf_state_dict_prefix="",
        ),
        "qwen3_5_moe": ModelEntry(
            macos_class=Qwen3_5MoeForCausalLM,
            hf_config_attr="text_config",
            # Empty prefix keeps the top-level untied `lm_head.weight`; the
            # loader still buckets `model.language_model.layers.*` per-layer.
            hf_state_dict_prefix="",
        ),
    }


# Type alias for the remapping dict
MODEL_TYPE_REMAPPING: dict[str, str] = {
    "gemma3": "gemma3_text",
    "qwen2_5": "qwen2",
}


def get_model_entry(model_type: str) -> ModelEntry:
    """Look up a model by HuggingFace model_type."""
    registry = _get_registry()
    remapped = MODEL_TYPE_REMAPPING.get(model_type, model_type)
    if remapped not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(f"Unknown model type '{model_type}'. Available: {available}")
    return registry[remapped]


def list_models() -> list[str]:
    """List all supported model types."""
    return sorted(_get_registry().keys())
