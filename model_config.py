"""
Model configuration and loading utilities.
"""

import json
import os
from pathlib import Path

from tinygrad import Context, Device, Tensor, dtypes, nn
from tinygrad.helpers import fetch
from tinygrad.nn.state import gguf_load, load_state_dict, safe_load, torch_load

from extra.bench_log import BenchEvent, WallTimeEvent
from extra.models.llama import convert_from_gguf, convert_from_huggingface, fix_bf16

MODEL_PARAMS = {
    "1B": {
        "args": {
            "dim": 2048,
            "n_heads": 32,
            "n_kv_heads": 8,
            "n_layers": 16,
            "norm_eps": 1e-5,
            "rope_theta": 500000,
            "vocab_size": 128256,
            "hidden_dim": 8192,
        },
        "files": 1,
    },
    "8B": {
        "args": {
            "dim": 4096,
            "n_heads": 32,
            "n_kv_heads": 8,
            "n_layers": 32,
            "norm_eps": 1e-5,
            "rope_theta": 500000,
            "vocab_size": 128256,
            "hidden_dim": 14336,
        },
        "files": 1,
    },
    "70B": {
        "args": {
            "dim": 8192,
            "n_heads": 64,
            "n_kv_heads": 8,
            "n_layers": 80,
            "norm_eps": 1e-5,
            "rope_theta": 500000,
            "vocab_size": 128256,
            "hidden_dim": 28672,
        },
        "files": 8,
    },
    "405B": {
        "args": {
            "dim": 16384,
            "n_heads": 128,
            "n_kv_heads": 8,
            "n_layers": 126,
            "norm_eps": 1e-5,
            "rope_theta": 500000,
            "vocab_size": 128256,
            "hidden_dim": 53248,
        },
        "files": 191,
    },
}


def concat_weights(models, device=None):
    def convert(name) -> Tensor:
        disk_tensors: list[Tensor] = [model[name] for model in models]
        if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
            return disk_tensors[0].to(device=device)
        axis = 1 if name.endswith((".attention.wo.weight", ".feed_forward.w2.weight")) else 0
        lazy_tensors = [data.to(device=device) for data in disk_tensors]
        return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)

    return {name: convert(name) for name in {name: None for model in models for name in model}}


def load_weights(fn: str):
    if fn.endswith(".index.json"):
        with open(fn) as fp:
            weight_map = json.load(fp)["weight_map"]
        parts = {n: load_weights(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
        return {k: parts[n][k] for k, n in weight_map.items()}
    if fn.endswith(".gguf"):
        gguf_tensor = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}").to(Device.DEFAULT)
        return gguf_load(gguf_tensor)[1]
    if fn.endswith(".safetensors"):
        return safe_load(fn)
    return torch_load(fn)


def build_transformer(
    model_path: Path,
    model_size="8B",
    quantize=None,
    scale_dtype=dtypes.float16,
    device=None,
    max_context=8192,
    load_weights_flag=True,
):
    from extra.models.llama import Transformer
    from quantization import Int8Embedding, Int8Linear, NF4Linear

    if quantize == "int8":
        linear, embedding, quantize_embeds = Int8Linear, Int8Embedding, True
    elif quantize == "nf4":
        linear, embedding, quantize_embeds = NF4Linear(64), nn.Embedding, False
    else:
        linear, embedding, quantize_embeds = nn.Linear, nn.Embedding, False

    model = Transformer(
        **MODEL_PARAMS[model_size]["args"], linear=linear, embedding=embedding, max_context=max_context, jit=True
    )

    if not load_weights_flag:
        return model

    with WallTimeEvent(BenchEvent.LOAD_WEIGHTS):
        if model_path.is_dir():
            if (model_path / "model.safetensors.index.json").exists():
                weights = load_weights(str(model_path / "model.safetensors.index.json"))
            elif (model_path / "model.safetensors").exists():
                weights = load_weights(str(model_path / "model.safetensors"))
            else:
                weights = concat_weights(
                    [
                        load_weights(str(model_path / f"consolidated.{i:02d}.pth"))
                        for i in range(MODEL_PARAMS[model_size]["files"])
                    ],
                    device[0] if isinstance(device, tuple) else device,
                )
        else:
            weights = load_weights(str(model_path))

        if "model.embed_tokens.weight" in weights:
            weights = convert_from_huggingface(
                weights,
                MODEL_PARAMS[model_size]["args"]["n_layers"],
                MODEL_PARAMS[model_size]["args"]["n_heads"],
                MODEL_PARAMS[model_size]["args"]["n_kv_heads"],
            )
        elif "token_embd.weight" in weights:
            weights = convert_from_gguf(weights, MODEL_PARAMS[model_size]["args"]["n_layers"])
        weights = fix_bf16(weights)

        with Context(BEAM=0):
            if quantize == "float16":
                weights = {k: v.cast(quantize).contiguous() for k, v in weights.items()}
            elif quantize is not None:
                weights = linear.quantize(weights, device, scale_dtype, quantize_embeds)
                for _, v in weights.items():
                    v.realize()

            if isinstance(device, tuple):
                for k, v in nn.state.get_state_dict(model).items():
                    if "scale" in k:
                        v.shard_(device, axis=None)
                    elif ".attention." in k:
                        v.shard_(device, axis=-1)
                    elif ".feed_forward.w1." in k or ".feed_forward.w3." in k:
                        v.shard_(device, axis=0)
                    elif ".feed_forward." in k:
                        v.shard_(device, axis=-1)
                    elif "tok_embeddings.weight" in k or "output.weight" in k:
                        v.shard_(device, axis=0)
                    else:
                        v.shard_(device, axis=None)

            load_state_dict(model, weights, strict=False, consume=True)
    return model


def find_model_in_default_dir(size: str) -> Path | None:
    """Look for model files in ~/models directory based on size"""
    models_dir = Path.home() / "models"

    size_to_dir = {
        "1B": "llama3-1b-instruct",
        "8B": "llama3-8b-sfr",
        "70B": "DeepSeek-R1-Distill-Llama-70B",
        "405B": "llama3-405b",
    }

    if size not in size_to_dir:
        return None

    model_dir = models_dir / size_to_dir[size]
    if not model_dir.exists():
        return None

    if size == "1B":
        gguf_file = model_dir / "Llama-3.2-1B-Instruct-Q6_K.gguf"
        if gguf_file.exists():
            return gguf_file
    elif size in ["8B", "70B"]:
        index_file = model_dir / "model.safetensors.index.json"
        if index_file.exists():
            return index_file

    return None


def download_model(size: str) -> Path:
    """Download model based on size"""
    if size == "1B":
        model_dir = os.path.expanduser("~/models/llama3-1b-instruct")
        os.makedirs(model_dir, exist_ok=True)
        fetch(
            "https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model",
            os.path.join(model_dir, "tokenizer.model"),
        )
        return fetch(
            "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
            os.path.join(model_dir, "Llama-3.2-1B-Instruct-Q6_K.gguf"),
        )
    if size == "8B":
        model_dir = os.path.expanduser("~/models/llama3-8b-sfr")
        os.makedirs(model_dir, exist_ok=True)
        fetch(
            "https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model",
            os.path.join(model_dir, "tokenizer.model"),
        )
        for i in range(1, 5):
            fetch(
                f"https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-{i:05d}-of-00004.safetensors",
                os.path.join(model_dir, f"model-{i:05d}-of-00004.safetensors"),
            )
        return fetch(
            "https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/raw/main/model.safetensors.index.json",
            os.path.join(model_dir, "model.safetensors.index.json"),
        )
    if size == "70B":
        model_dir = os.path.expanduser("~/models/DeepSeek-R1-Distill-Llama-70B")
        os.makedirs(model_dir, exist_ok=True)
        model_path = fetch(
            "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/model.safetensors.index.json?download=true",
            os.path.join(model_dir, "model.safetensors.index.json"),
        )
        fetch(
            "https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model",
            os.path.join(model_dir, "tokenizer.model"),
        )
        for i in range(17):
            fetch(
                f"https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/model-{i + 1:05d}-of-000017.safetensors?download=true",
                os.path.join(model_dir, f"model-{i + 1:05d}-of-000017.safetensors"),
            )
        return model_path
    raise ValueError(f"Unsupported model size: {size}")


def resolve_model_path(model_path: Path | None, size: str, download: bool) -> Path:
    """Resolve model path, downloading if necessary"""
    if not model_path and not download:
        found_model = find_model_in_default_dir(size)
        if found_model:
            print(f"Found model: {found_model}")
            return found_model
        print(f"No {size} model found in ~/models, will download...")
        download = True

    if download or not model_path:
        return download_model(size)

    return model_path
