"""Persistent reference-model worker. Stdout is reserved for JSON protocol."""
import base64
import io
import json
import os
import sys

sys.stdout = protocol = sys.stdout

def send(**data):
    protocol.write(json.dumps(data, separators=(",", ":")) + "\n")
    protocol.flush()

try:
    import torch
    from diffusers import AutoModel, BriaFibo2Pipeline
    from huggingface_hub import hf_hub_download
    from PIL import Image

    device = os.environ.get("RMBG_DEVICE", "cuda")
    if device not in ("cuda", "cpu", "mps"):
        raise ValueError("RMBG_DEVICE must be cuda, cpu, or mps")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model_id = os.environ.get("RMBG_MODEL_ID", "briaai/RMBG-3.0")
    base_model_id = os.environ.get("RMBG_BASE_MODEL_ID", "briaai/Fibo-2.0-edit-turbo")
    transformer = AutoModel.from_pretrained(model_id, subfolder="transformer", dtype=dtype)
    pipe = BriaFibo2Pipeline.from_pretrained(
        base_model_id, transformer=transformer, dtype=dtype
    )
    if device == "cuda":
        # Sequential offload trades speed for the ability to run on a 24 GiB card.
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)
    fonts = [hf_hub_download(base_model_id, n, subfolder="fonts")
             for n in ("Aileron-Regular.otf", "DejaVuSans.ttf")]
    instruction = ("Generate a detailed grayscale alpha matte. Map the opaque foreground to white and the "
                   "background to black. Produce soft, anti-aliased grayscale gradients at the edges of the "
                   "subject to represent fine details and transparency.")
    prompt = json.dumps({"edit_instruction": instruction})
    send(ready=True)
except Exception as exc:
    message = str(exc)
    if any(hint in message.lower() for hint in ("gated", "restricted", "401", "access to model")):
        message = ("Gated model access denied. Accept the access forms for briaai/RMBG-3.0 "
                   "and briaai/Fibo-2.0-edit-turbo, then set HF_TOKEN in your environment.")
    send(error=message)
    sys.exit(1)

for line in sys.stdin:
    try:
        request = json.loads(line)
        source = Image.open(io.BytesIO(base64.b64decode(request["image"]))).convert("RGB")
        generator = torch.Generator(device="cpu" if device == "cpu" else device).manual_seed(0)
        result = pipe(prompt=prompt, image=source, num_inference_steps=4,
                      guidance_scale=1.0, generator=generator, fonts=fonts).images[0]
        matte = result.convert("L")
        buffer = io.BytesIO()
        matte.save(buffer, format="PNG")
        send(matte=base64.b64encode(buffer.getvalue()).decode("ascii"))
    except Exception as exc:
        send(error=str(exc))
