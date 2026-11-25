"""Fetch Bria's official INT8 ComfyUI bundle with Hub's resumable cache."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download, model_info

REPO = "briaai/RMBG-3.0-Comfy"
QWEN_REPO = "Qwen/Qwen3-VL-4B-Instruct"
FILES = (
    "diffusion_models/rmbg3_int8_convrot.safetensors",
    "text_encoders/qwen3vl_4b_int8_convrot.safetensors",
    "vae/flux2-vae.safetensors",
)
QWEN_FILES = ("config.json", "tokenizer.json")

def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: fetch_comfy.py COMFYUI_MODELS_DIR")
    destination = Path(sys.argv[1]).resolve()
    details = model_info(REPO, files_metadata=True)
    sizes = {file.rfilename: file.size for file in details.siblings}

    def fetch(filename):
        if filename not in sizes or not sizes[filename]:
            raise RuntimeError(f"missing size metadata for {filename}")
        print(f"Fetching {filename} ({sizes[filename]} bytes)", flush=True)
        downloaded = Path(hf_hub_download(REPO, filename=filename, local_dir=destination,
                                          token=False))
        if downloaded.stat().st_size != sizes[filename]:
            downloaded.unlink(missing_ok=True)
            raise RuntimeError(f"incomplete download for {filename}")
        print(f"Complete {filename}", flush=True)

    def fetch_qwen(filename):
        print(f"Fetching Qwen metadata {filename}", flush=True)
        downloaded = Path(hf_hub_download(
            QWEN_REPO,
            filename=filename,
            local_dir=destination / "qwen3vl",
            token=False,
        ))
        print(f"Complete Qwen metadata {downloaded.name}", flush=True)

    with ThreadPoolExecutor(max_workers=len(FILES) + len(QWEN_FILES)) as pool:
        pending = [pool.submit(fetch, filename) for filename in FILES]
        pending.extend(pool.submit(fetch_qwen, filename) for filename in QWEN_FILES)
        for task in as_completed(pending):
            task.result()
    print("Quantized files and Qwen metadata ready.", flush=True)

if __name__ == "__main__":
    main()
