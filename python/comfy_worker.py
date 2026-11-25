"""Persistent ComfyUI API bridge for Bria's official INT8 ConvRot checkpoint.

Requires a running ComfyUI with 1038lab/ComfyUI-RMBG's AILab_RMBG3 node.
Only stdlib dependencies. Stdout is reserved for the line JSON protocol.
"""
import base64
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid

base = os.environ.get("RMBG_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")

def send(**data):
    sys.stdout.write(json.dumps(data, separators=(",", ":")) + "\n")
    sys.stdout.flush()

def api(path, data=None, headers=None):
    request = urllib.request.Request(base + path, data=data, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()

def json_api(path, value=None):
    data = None if value is None else json.dumps(value).encode("utf-8")
    return json.loads(api(path, data, {"Content-Type": "application/json"} if data else None))

def upload(png):
    boundary = "----rembg" + uuid.uuid4().hex
    filename = "rembg_" + uuid.uuid4().hex + ".png"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
            "Content-Type: image/png\r\n\r\n").encode() + png + (f"\r\n--{boundary}--\r\n").encode()
    result = json.loads(api("/upload/image", body, {"Content-Type": "multipart/form-data; boundary=" + boundary}))
    return result["name"]

def run(png):
    filename = upload(png)
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": filename}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "rmbg3_int8_convrot.safetensors", "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_int8_convrot.safetensors", "type": "fibo2", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "5": {"class_type": "AILab_RMBG3", "inputs": {"model": ["2", 0], "clip": ["3", 0], "vae": ["4", 0], "image": ["1", 0], "steps": 4, "seed": 0}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 2], "filename_prefix": "rembg_rs_matte"}},
    }
    submitted = json_api("/prompt", {"prompt": graph, "client_id": "rembg-rs"})
    if "error" in submitted:
        raise RuntimeError(submitted["error"])
    prompt_id = submitted["prompt_id"]
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        history = json_api("/history/" + urllib.parse.quote(prompt_id))
        if prompt_id in history:
            job = history[prompt_id]
            if job.get("status", {}).get("status_str") == "error":
                raise RuntimeError("ComfyUI workflow failed; inspect ComfyUI logs")
            images = job.get("outputs", {}).get("6", {}).get("images", [])
            if not images:
                raise RuntimeError("ComfyUI returned no matte image")
            item = images[0]
            query = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")})
            return api("/view?" + query)
        time.sleep(0.5)
    raise TimeoutError("ComfyUI workflow did not finish within 10 minutes")

try:
    nodes = json_api("/object_info")
    needed = ("AILab_RMBG3", "UNETLoader", "CLIPLoader", "VAELoader")
    missing = [name for name in needed if name not in nodes]
    if missing:
        raise RuntimeError("ComfyUI missing nodes: " + ", ".join(missing))
    send(ready=True)
except Exception as exc:
    send(error=str(exc))
    sys.exit(1)

for line in sys.stdin:
    try:
        request = json.loads(line)
        matte = run(base64.b64decode(request["image"]))
        send(matte=base64.b64encode(matte).decode("ascii"))
    except Exception as exc:
        send(error=str(exc))
