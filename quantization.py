import json
import shutil
from pathlib import Path

import mlx.core as mx

SRC = Path("model")
DST = Path("model-int4")
GROUP_SIZE = 64
BITS = 4


def quantize_shard(path):
    out = {}
    for name, w in mx.load(str(path)).items():
        if w.ndim == 2 and name.endswith(".weight"):
            base = name[: -len(".weight")]
            q, scales, biases = mx.quantize(w.astype(mx.bfloat16), group_size=GROUP_SIZE, bits=BITS)
            out[f"{base}.weight"], out[f"{base}.scales"], out[f"{base}.biases"] = q, scales, biases
        else:
            out[name] = w.astype(mx.bfloat16)
    return out


def main():
    DST.mkdir(exist_ok=True)
    weight_map = {}

    for shard in sorted(SRC.glob("model-*.safetensors")):
        tensors = quantize_shard(shard)
        mx.save_safetensors(str(DST / shard.name), tensors, metadata={"format": "mlx"})
        weight_map.update({k: shard.name for k in tensors})
        print(f"{shard.name}: {len(tensors)} tensors")

    (DST / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}, indent=2))

    config = json.loads((SRC / "config.json").read_text())
    config["quantization"] = {"group_size": GROUP_SIZE, "bits": BITS}
    (DST / "config.json").write_text(json.dumps(config, indent=2))

    for f in ["generation_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        shutil.copy(SRC / f, DST / f)


if __name__ == "__main__":
    main()
