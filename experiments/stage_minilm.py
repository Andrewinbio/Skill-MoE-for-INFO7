"""Download the pinned upstream keyword-mapping encoder on a networked host."""
import hashlib
import json
from pathlib import Path
import subprocess

REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FILES = ["config.json", "config_sentence_transformers.json", "modules.json",
         "1_Pooling/config.json", "sentence_bert_config.json", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"]


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "model-cache" / "all-MiniLM-L6-v2"
    hashes = {}
    for name in FILES:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            url = f"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{REVISION}/{name}"
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--retry", "2", "--max-time", "180", "--output", str(temporary), url], check=True)
            temporary.replace(path)
        if name.endswith('.json'):
            json.loads(path.read_text())
        if name == 'model.safetensors' and path.stat().st_size < 80_000_000:
            raise ValueError('Incomplete or pointer-only weights')
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (target / "source-manifest.json").write_text(json.dumps({
        "model": "sentence-transformers/all-MiniLM-L6-v2", "revision": REVISION,
        "files_sha256": hashes,
    }, indent=2) + '\n')
    print(target)
