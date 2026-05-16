#!/usr/bin/env python
"""Average PEFT LoRA adapter weights into a reusable initialization adapter."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_dirs = [Path(path) for path in args.adapter_dirs]
    if not adapter_dirs:
        raise ValueError("at least one adapter dir is required")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    states = []
    for adapter_dir in adapter_dirs:
        weights_path = adapter_dir / "adapter_model.safetensors"
        config_path = adapter_dir / "adapter_config.json"
        if not weights_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"missing PEFT adapter files under {adapter_dir}")
        states.append(load_file(str(weights_path), device="cpu"))

    keys = set(states[0])
    for adapter_dir, state in zip(adapter_dirs[1:], states[1:], strict=True):
        if set(state) != keys:
            raise ValueError(f"adapter key mismatch for {adapter_dir}")

    averaged = {}
    for key in sorted(keys):
        tensors = [state[key].to(torch.float32) for state in states]
        averaged[key] = torch.stack(tensors, dim=0).mean(dim=0).to(states[0][key].dtype)

    save_file(averaged, str(output_dir / "adapter_model.safetensors"))
    shutil.copy2(adapter_dirs[0] / "adapter_config.json", output_dir / "adapter_config.json")
    readme = output_dir / "README.md"
    readme.write_text(
        "Averaged LoRA adapter generated from source query adapters.\n",
        encoding="utf-8",
    )
    (output_dir / "average_summary.json").write_text(
        json.dumps(
            {
                "adapter_count": len(adapter_dirs),
                "adapter_dirs": [str(path) for path in adapter_dirs],
                "output_dir": str(output_dir),
                "weight_keys": len(keys),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "adapter_count": len(adapter_dirs)}, sort_keys=True))


if __name__ == "__main__":
    main()
