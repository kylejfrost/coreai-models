# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Create model bundles from exported .aimodel files."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

METADATA_VERSION = "0.2"


def bundle_llm_asset(
    bundle_path: Path,
    hf_model_id: str,
    hf_config: Any,
    compression: str,
    name: str,
    decode_asset_name: str | None = None,
) -> None:
    """Add tokenizer and metadata.json (0.2 schema) to an LLM bundle.

    Expects ``{name}.aimodel`` to already exist inside bundle_path.
    """
    _write_tokenizer(bundle_path / "tokenizer", hf_model_id)
    _write_metadata(bundle_path, hf_model_id, hf_config, compression, name, decode_asset_name)


def _write_tokenizer(dest: Path, hf_model_id: str) -> None:
    logger.info(f"Saving tokenizer from {hf_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    tokenizer.save_pretrained(str(dest))


def _write_metadata(
    bundle_path: Path,
    hf_model_id: str,
    hf_config: Any,
    compression: str,
    name: str,
    decode_asset_name: str | None = None,
) -> None:
    assets = {"main": f"{name}.aimodel"}
    if decode_asset_name is not None:
        assets["decode"] = decode_asset_name
    function_map = {"main": ["main"]}
    if decode_asset_name is not None or os.environ.get(
        "COREAI_MACOS_DECODE_ENTRYPOINT", ""
    ).lower() in ("1", "true", "yes"):
        function_map["decode"] = ["decode"]
    metadata: dict[str, Any] = {
        "metadata_version": METADATA_VERSION,
        "kind": "llm",
        "name": name,
        "assets": assets,
        "language": {
            "tokenizer": hf_model_id,
            "vocab_size": getattr(hf_config, "vocab_size", None),
            "max_context_length": getattr(hf_config, "max_position_embeddings", None),
            "embedded_tokenizer": True,
            "function_map": function_map,
        },
        "source": {
            "model_definition": "torch",
            "hf_model_id": hf_model_id,
        },
        "compression": compression if compression != "none" else None,
        "compilation": {
            "date": datetime.now().astimezone().isoformat(),
            "targets": [],
        },
    }
    metadata_path = bundle_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote metadata to {metadata_path}")
