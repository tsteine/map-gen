#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import safetensors.torch
from safetensors import safe_open


TRAINING_CHECKPOINT_FORMAT = "map-gen-training-session-checkpoint-v3"
MODEL_EXPORT_FORMAT = "map-gen-model-export-v1"
MODEL_PREFIXES = ("ema_model", "balance_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the EMA model and balance model weights from a training checkpoint "
            "to a smaller safetensors file."
        ),
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Training checkpoint safetensors file to export from.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output safetensors file containing only ema_model.* and balance_model.* tensors.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def validate_checkpoint_metadata(path: Path, metadata: dict[str, str] | None) -> dict[str, str]:
    if metadata is None:
        raise ValueError(f"checkpoint metadata missing in {path}")
    if metadata["format"] != TRAINING_CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format in {path}")
    for field in ("config", "num_episodes", "aim_run_hash"):
        if field not in metadata:
            raise ValueError(f"checkpoint metadata field {field!r} missing in {path}")
    return metadata


def model_prefix(name: str) -> str | None:
    for prefix in MODEL_PREFIXES:
        if name.startswith(f"{prefix}."):
            return prefix
    return None


def export_model_tensors(checkpoint_path: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output file already exists: {output_path}")

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        checkpoint_metadata = validate_checkpoint_metadata(
            checkpoint_path,
            checkpoint.metadata(),
        )
        tensors = {}
        tensor_counts = dict.fromkeys(MODEL_PREFIXES, 0)
        for name in checkpoint.keys():
            prefix = model_prefix(name)
            if prefix is None:
                continue
            tensors[name] = checkpoint.get_tensor(name)
            tensor_counts[prefix] += 1

    missing_prefixes = [prefix for prefix, count in tensor_counts.items() if count == 0]
    if missing_prefixes:
        raise ValueError(
            f"checkpoint missing model tensor group(s): {', '.join(missing_prefixes)}"
        )

    metadata = {
        "format": MODEL_EXPORT_FORMAT,
        "config": checkpoint_metadata["config"],
        "source_format": checkpoint_metadata["format"],
        "source_num_episodes": checkpoint_metadata["num_episodes"],
        "source_aim_run_hash": checkpoint_metadata["aim_run_hash"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    safetensors.torch.save_file(tensors, temp_path, metadata=metadata)
    os.replace(temp_path, output_path)

    print(
        f"Exported {len(tensors)} tensor(s) from {checkpoint_path} to {output_path}: "
        f"{', '.join(f'{prefix}={tensor_counts[prefix]}' for prefix in MODEL_PREFIXES)}"
    )


def main() -> int:
    args = parse_args()
    export_model_tensors(args.checkpoint, args.output, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
