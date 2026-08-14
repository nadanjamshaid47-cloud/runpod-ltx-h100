#!/usr/bin/env python3
import os, time, uuid, runpod, requests

COMFY_URL = "http://127.0.0.1:8188"

def build_workflow(job_input):
    prompt_text = job_input.get("prompt", "")
    negative = job_input.get("negative", "blurry, low quality, distorted")
    width = job_input.get("width", 768)
    height = job_input.get("height", 512)
    length = job_input.get("length", 65)
    steps = job_input.get("steps", 20)
    seed = job_input.get("seed", -1)
    if seed < 0:
        seed = int(time.time()) % 100000

    return {
        "1": {"inputs": {"ckpt_name": "ltx-2.3-22b-dev.safetensors", "weight_dtype": "fp8_e4m3fn"}, "class_type": "UNETLoader"},
        "2": {"inputs": {"gemma_path": "gemma_3_12B_it_fp4_mixed.safetensors", "ltxv_path": "ltx-2.3-22b-dev.safetensors", "max_length": 128}, "class_type": "LTXVGemmaCLIPModelLoader"},
        "3": {"inputs": {"vae_name": "ltx-video-2b-v0.9.safetensors"}, "class_type": "VAELoader"},
        "4": {"inputs": {"text": prompt_text, "clip": ["2", 0]}, "class_type": "CLIPTextEncode"},
        "5": {"inputs": {"text": negative, "clip": ["2", 0]}, "class_type": "CLIPTextEncode"},
        "6": {"inputs": {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": 25.0}, "class_type": "LTXVConditioning"},
        "7": {"inputs": {"width": width, "height": height, "length": length, "batch_size": 1}, "class_type": "EmptyLTXVLatentVideo"},
        "8": {"inputs": {"model": ["1", 0], "max_shift": 3.0, "base_shift": 0.5}, "class_type": "ModelSamplingLTXV"},
        "9": {"inputs": {"seed": seed, "steps": steps, "cfg": 3.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": ["8", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["7", 0]}, "class_type": "KSampler"},
        "10": {"inputs": {"samples": ["9", 0], "vae": ["3", 0]}, "class_type": "VAEDecode"},
        "11": {"inputs": {"filename_prefix": f"ltx_{uuid.uuid4().hex[:6]}", "fps": 24, "lossless": False, "quality": 85, "method": "default", "images": ["10", 0]}, "class_type": "SaveAnimatedWEBP"}
    }

def generate(workflow):
    client_id = str(uuid.uuid4())
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=10)
    data = r.json()
    if "error" in data:
        return {"status": "error", "error": data["error"]}
    pid = data["prompt_id"]

    for _ in range(120):
        time.sleep(5)
        h = requests.get(f"{COMFY_URL}/history/{pid}", timeout=10).json()
        if pid in h:
            if h[pid].get("status", {}).get("status_str") == "error":
                return {"status": "error"}
            if h[pid].get("outputs"):
                return {"status": "completed", "outputs": h[pid]["outputs"]}
    return {"status": "timeout"}

def handler(job):
    start = time.time()
    result = generate(build_workflow(job["input"]))
    return {"status": result["status"], "time": round(time.time()-start, 2), "outputs": result.get("outputs", {})}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
