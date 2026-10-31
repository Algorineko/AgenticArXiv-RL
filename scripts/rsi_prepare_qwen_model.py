"""Download and fingerprint the public Qwen3-8B snapshot on temporary storage."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    if args.manifest.exists():
        raise SystemExit("Manifest already exists; refusing to overwrite")
    if shutil.disk_usage(args.model_dir.parent).free < 45 * 1024**3:
        raise SystemExit("Less than 45 GiB free before model download")
    from modelscope import snapshot_download
    snapshot_download("Qwen/Qwen3-8B", revision="master", local_dir=str(args.model_dir),
                      cache_dir=str(args.cache_dir), max_workers=2,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    config = json.loads((args.model_dir / "config.json").read_text())
    weights = sorted(args.model_dir.glob("*.safetensors"))
    if config.get("model_type") != "qwen3" or not weights or sum(p.stat().st_size for p in weights) < 15_000_000_000:
        raise SystemExit("Downloaded files are incomplete or have the wrong model type")
    selected = sorted(set(weights + [p for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                                                     "model.safetensors.index.json")
                                      if (p := args.model_dir / name).exists()]))
    manifest = {"model_id": "Qwen/Qwen3-8B", "source": "https://modelscope.cn/models/Qwen/Qwen3-8B",
                "requested_revision": "master", "resolved_revision": "not provided; file hashes pin downloaded content",
                "model_type": config["model_type"], "modelscope_version": importlib.metadata.version("modelscope"),
                "files": {p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in selected},
                "source_sha256": file_sha256(Path(__file__)),
                "storage": "temporary server filesystem; not a durable backup",
                "gpu_used_for_download": False}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"model_id": manifest["model_id"], "weight_files": len(weights),
                      "weight_bytes": sum(p.stat().st_size for p in weights), "manifest": str(args.manifest)}), flush=True)


if __name__ == "__main__":
    main()
