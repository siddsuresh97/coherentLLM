"""Minimal inference-only LoRA application for PEFT adapter checkpoints.

The coherence env used for these follow-up jobs does not include PEFT, but the
adapters are standard LoRA safetensors plus adapter_config.json. This module
installs the low-rank updates directly on matching torch.nn.Linear modules.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable

import torch
from safetensors.torch import load_file, safe_open


LORA_PREFIXES = ("base_model.model.",)


class LoRAWeights(torch.nn.Module):
    def __init__(self, a: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.register_buffer("a", a.contiguous(), persistent=False)
        self.register_buffer("b", b.contiguous(), persistent=False)


class MultiLoRALinear(torch.nn.Module):
    """Wrap a Linear layer and add one selected LoRA adapter at inference time."""

    def __init__(self, base: torch.nn.Linear):
        super().__init__()
        self.base = base
        self.adapters = torch.nn.ModuleDict()
        self.scalings: Dict[str, float] = {}
        self.active_adapter: str | None = None

    def add_adapter(
        self,
        name: str,
        a: torch.Tensor,
        b: torch.Tensor,
        scaling: float,
    ) -> None:
        self.adapters[name] = LoRAWeights(a, b)
        self.scalings[name] = float(scaling)

    def set_active(self, name: str | None) -> None:
        if name is not None and name not in self.adapters:
            raise KeyError(f"adapter {name!r} is not installed on this module")
        self.active_adapter = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.active_adapter is None:
            return y
        weights = self.adapters[self.active_adapter]
        # LoRA update: y += scaling * (x A^T) B^T.
        update = torch.nn.functional.linear(
            torch.nn.functional.linear(x.to(weights.a.dtype), weights.a),
            weights.b,
        )
        return y + update.to(y.dtype) * self.scalings[self.active_adapter]


def adapter_config(adapter_dir: str | Path) -> dict:
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    with cfg_path.open() as f:
        return json.load(f)


def adapter_sha256(adapter_dir: str | Path) -> str:
    h = hashlib.sha256()
    with (Path(adapter_dir) / "adapter_model.safetensors").open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lora_scaling(config: dict) -> float:
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    if config.get("use_rslora"):
        return alpha / (rank ** 0.5)
    return alpha / rank


def _strip_prefix(module_name: str) -> str:
    for prefix in LORA_PREFIXES:
        if module_name.startswith(prefix):
            return module_name[len(prefix):]
    return module_name


def _split_lora_key(key: str) -> tuple[str, str] | None:
    parts = key.split(".")
    for idx, part in enumerate(parts):
        if part in {"lora_A", "lora_B"}:
            which = "A" if part == "lora_A" else "B"
            return _strip_prefix(".".join(parts[:idx])), which
    return None


def lora_tensor_shapes(adapter_dir: str | Path) -> list[dict]:
    rows = []
    path = Path(adapter_dir) / "adapter_model.safetensors"
    with safe_open(path, framework="pt", device="cpu") as f:
        grouped: dict[str, dict[str, tuple[int, ...]]] = {}
        for key in f.keys():
            split = _split_lora_key(key)
            if split is None:
                continue
            module_name, which = split
            grouped.setdefault(module_name, {})[which] = tuple(f.get_tensor(key).shape)
    for module_name in sorted(grouped):
        pair = grouped[module_name]
        if "A" not in pair or "B" not in pair:
            continue
        a_shape = pair["A"]
        b_shape = pair["B"]
        rows.append(
            {
                "module": module_name,
                "lora_A_shape": list(a_shape),
                "lora_B_shape": list(b_shape),
                "delta_shape": [b_shape[0], a_shape[1]],
            }
        )
    return rows


def _get_parent(model: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, str]:
    parent_name, child_name = module_name.rsplit(".", 1)
    return model.get_submodule(parent_name), child_name


def install_lora_adapter(
    model: torch.nn.Module,
    adapter_dir: str | Path,
    adapter_name: str,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    activate: bool = True,
) -> int:
    """Install one adapter on a model and return the number of wrapped modules."""
    adapter_dir = Path(adapter_dir)
    config = adapter_config(adapter_dir)
    scaling = lora_scaling(config)
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        split = _split_lora_key(key)
        if split is None:
            continue
        module_name, which = split
        grouped.setdefault(module_name, {})[which] = tensor

    installed = 0
    for module_name, pair in sorted(grouped.items()):
        if "A" not in pair or "B" not in pair:
            raise ValueError(f"incomplete LoRA tensor pair for {module_name}")
        parent, child_name = _get_parent(model, module_name)
        child = getattr(parent, child_name)
        if isinstance(child, MultiLoRALinear):
            wrapped = child
        else:
            if not isinstance(child, torch.nn.Linear):
                raise TypeError(f"{module_name} is {type(child)}, expected Linear")
            wrapped = MultiLoRALinear(child)
            setattr(parent, child_name, wrapped)

        target_device = device if device is not None else wrapped.base.weight.device
        target_dtype = dtype if dtype is not None else wrapped.base.weight.dtype
        wrapped.add_adapter(
            adapter_name,
            pair["A"].to(device=target_device, dtype=target_dtype),
            pair["B"].to(device=target_device, dtype=target_dtype),
            scaling,
        )
        if activate:
            wrapped.set_active(adapter_name)
        installed += 1
    return installed


def set_active_lora(model: torch.nn.Module, adapter_name: str | None) -> None:
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            module.set_active(adapter_name)


def installed_lora_modules(model: torch.nn.Module) -> Iterable[MultiLoRALinear]:
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            yield module
