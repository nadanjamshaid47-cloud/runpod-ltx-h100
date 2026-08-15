#!/usr/bin/env python3
"""
LTX Video 2.3 — FINAL: 3 Plans
  basic     → FREE  | 1 worker, direct generate, 854×480, 8 sec
  standard  → PAID  | 3 workers parallel, 1280×720, ~14 sec
  pro       → PAID  | 5 workers parallel, 1920×1080, ~23 sec

Basic saves money: NO master dispatch, generates directly on 1 worker.
Standard/Pro use parallel workers for speed.
"""

import os, time, uuid, base64, asyncio, aiohttp, subprocess, re
import runpod, requests

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "/app/ComfyUI/output"

# ═══════════════════════════════════════════════════════════
# 3 PLANS
# ═══════════════════════════════════════════════════════════
PLANS = {
    "basic": {
        "chunks": 1,
        "width": 854,
        "height": 480,
        "length": 193,      # ~7.7 sec
        "steps": 20,
        "cfg": 3.0,
        "sampler": "euler_ancestral",
        "paid": False,      # FREE — direct generate, no parallel
    },
    "standard": {
        "chunks": 3,
        "width": 1280,
        "height": 720,
        "length": 121,      # ~4.8 sec per chunk
        "steps": 20,
        "cfg": 3.0,
        "sampler": "euler_ancestral",
        "paid": True,       # PAID — parallel workers
    },
    "pro": {
        "chunks": 5,
        "width": 1920,
        "height": 1080,
        "length": 121,      # ~4.8 sec per chunk
        "steps": 25,
        "cfg": 3.5,
        "sampler": "euler_ancestral",
        "paid": True,       # PAID — parallel workers
    },
}

CHUNK_DURATION = 4.8
CROSSFADE = 0.5

# ═══════════════════════════════════════════════════════════
# STYLE PRESETS
# ═══════════════════════════════════════════════════════════
STYLES = {
    "cinematic": {
        "prefix": "cinematic film still, dramatic lighting, 35mm grain, smooth camera movement, ",
        "suffix": ", highly detailed, 8k, masterpiece",
        "negative": "blurry, low quality, distorted, ugly, watermark, text, logo, amateur, shaky"
    },
    "pixar": {
        "prefix": "Pixar 3D animation, Disney render, subsurface scattering, vibrant colors, smooth animation, ",
        "suffix": ", 3D CGI, global illumination, masterpiece",
        "negative": "blurry, low quality, distorted, ugly, 2d, flat, sketch, realistic, live action, shaky"
    },
    "anime": {
        "prefix": "anime style, Studio Ghibli, Makoto Shinkai, cel shaded, smooth animation, ",
        "suffix": ", masterpiece, ultra detailed anime",
        "negative": "blurry, low quality, distorted, ugly, 3d, realistic, photograph, shaky"
    },
    "realistic": {
        "prefix": "photorealistic, hyperrealistic, DSLR, natural lighting, smooth camera, ",
        "suffix": ", 8k uhd, sharp focus, masterpiece",
        "negative": "blurry, low quality, distorted, ugly, painting, drawing, cartoon, anime, shaky"
    },
    "cyberpunk": {
        "prefix": "cyberpunk, neon lights, futuristic, rain, reflections, smooth camera, ",
        "suffix": ", highly detailed, 8k, neon glow, masterpiece",
        "negative": "blurry, low quality, distorted, ugly, daytime, sunny, natural, shaky"
    },
    "none": {
        "prefix": "",
        "suffix": "",
        "negative": "blurry, low quality, distorted, ugly, watermark, text, logo, shaky"
    }
}


def enhance_prompt(prompt, style="cinematic"):
    preset = STYLES.get(style, STYLES["cinematic"])
    prompt = prompt.strip()
    if not prompt.endswith((".", "!", "?", ",")):
        prompt += ","
    enhanced = f"{preset['prefix']}{prompt}{preset['suffix']}"
    enhanced = re.sub(r'\s+', ' ', enhanced)
    enhanced = re.sub(r',+', ',', enhanced)
    enhanced = re.sub(r'\s+,', ',', enhanced)
    return enhanced.strip(', '), preset["negative"]


# ═══════════════════════════════════════════════════════════
# WORKFLOW
# ═══════════════════════════════════════════════════════════
def build_workflow(job_input, chunk_idx):
    prompt = job_input.get("prompt", "")
    negative = job_input.get("negative", "")
    style = job_input.get("style", "cinematic")
    auto_enhance = job_input.get("auto_enhance", True)
    width = job_input.get("width", 854)
    height = job_input.get("height", 480)
    length = job_input.get("length", 121)
    steps = job_input.get("steps", 20)
    cfg = job_input.get("cfg", 3.0)
    sampler = job_input.get("sampler", "euler_ancestral")
    seed = job_input.get("seed", -1)

    if auto_enhance:
        prompt, style_neg = enhance_prompt(prompt, style)
        negative = f"{negative}, {style_neg}" if negative else style_neg

    if seed < 0:
        seed = int(time.time()) % 100000
    chunk_seed = seed + chunk_idx * 1000

    return {
        "1": {"inputs": {"ckpt_name": "ltx-2.3-22b-dev.safetensors", "weight_dtype": "fp8_e4m3fn"}, "class_type": "UNETLoader"},
        "2": {"inputs": {"gemma_path": "gemma_3_12B_it_fp4_mixed.safetensors", "ltxv_path": "ltx-2.3-22b-dev.safetensors", "max_length": 128}, "class_type": "LTXVGemmaCLIPModelLoader"},
        "3": {"inputs": {"vae_name": "ltx-video-2b-v0.9.safetensors"}, "class_type": "VAELoader"},
        "4": {"inputs": {"text": prompt, "clip": ["2", 0]}, "class_type": "CLIPTextEncode"},
        "5": {"inputs": {"text": negative, "clip": ["2", 0]}, "class_type": "CLIPTextEncode"},
        "6": {"inputs": {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": 25.0}, "class_type": "LTXVConditioning"},
        "7": {"inputs": {"width": width, "height": height, "length": length, "batch_size": 1}, "class_type": "EmptyLTXVLatentVideo"},
        "8": {"inputs": {"model": ["1", 0], "max_shift": 3.0, "base_shift": 0.5}, "class_type": "ModelSamplingLTXV"},
        "9": {"inputs": {"seed": chunk_seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": "normal", "denoise": 1.0, "model": ["8", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["7", 0]}, "class_type": "KSampler"},
        "10": {"inputs": {"samples": ["9", 0], "vae": ["3", 0]}, "class_type": "VAEDecode"},
        "11": {"inputs": {"filename_prefix": f"chk_{chunk_idx:02d}", "fps": 25, "lossless": False, "quality": 85, "method": "default", "images": ["10", 0]}, "class_type": "SaveAnimatedWEBP"}
    }


def generate_chunk(workflow):
    """Generate one chunk"""
    client_id = str(uuid.uuid4())
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=10)
    data = r.json()
    if "error" in data:
        return {"status": "error", "error": data["error"]}

    pid = data["prompt_id"]
    for _ in range(300):
        time.sleep(3)
        h = requests.get(f"{COMFY_URL}/history/{pid}", timeout=10).json()
        if pid in h:
            status = h[pid].get("status", {})
            if status.get("status_str") == "error":
                return {"status": "error", "error": "ComfyUI error"}
            outputs = h[pid].get("outputs", {})
            if outputs and "11" in outputs:
                images = outputs["11"].get("images", [])
                if images:
                    fn = images[0]["filename"]
                    sf = images[0].get("subfolder", "")
                    fp = os.path.join(OUTPUT_DIR, sf, fn)
                    for _ in range(30):
                        if os.path.exists(fp):
                            break
                        time.sleep(1)
                    if os.path.exists(fp):
                        with open(fp, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        return {"status": "completed", "video_base64": b64, "filename": fn, "size_mb": round(os.path.getsize(fp)/(1024*1024), 2)}
    return {"status": "timeout"}


# ═══════════════════════════════════════════════════════════
# STITCH
# ═══════════════════════════════════════════════════════════
def stitch_chunks(chunk_files, output_path):
    n = len(chunk_files)
    if n == 0:
        raise ValueError("No chunks")
    if n == 1:
        subprocess.run(["ffmpeg", "-y", "-i", chunk_files[0], "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    mp4_files = []
    for i, src in enumerate(chunk_files):
        mp4 = f"/tmp/st_{i:02d}.mp4"
        subprocess.run(["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-r", "25", mp4],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mp4_files.append(mp4)

    filter_parts = []
    prev_label = "[0:v]"
    for i in range(1, n):
        offset = i * (CHUNK_DURATION - CROSSFADE)
        out_label = f"[v{i}]" if i < n - 1 else "[outv]"
        filter_parts.append(f"{prev_label}[{i}:v]xfade=transition=fade:duration={CROSSFADE}:offset={offset:.1f}{out_label}")
        prev_label = out_label

    inputs = [x for f in mp4_files for x in ("-i", f)]
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", ";".join(filter_parts), "-map", "[outv]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-r", "25", "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in mp4_files:
        os.remove(f)


# ═══════════════════════════════════════════════════════════
# MASTER: Parallel dispatch (for PAID plans only)
# ═══════════════════════════════════════════════════════════
async def submit_chunk(session, endpoint_id, api_key, job_input, chunk_idx, total):
    payload = {"input": {**job_input, "chunk_index": chunk_idx, "total_chunks": total}}
    async with session.post(f"https://api.runpod.ai/v2/{endpoint_id}/run",
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                            json=payload) as resp:
        data = await resp.json()
        return {"chunk": chunk_idx, "job_id": data.get("id")}


async def wait_chunk(session, endpoint_id, api_key, job_id, chunk_idx):
    for _ in range(300):
        async with session.get(f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}",
                               headers={"Authorization": f"Bearer {api_key}"}) as resp:
            status = await resp.json()
            st = status.get("status")
            if st == "COMPLETED":
                return status
            elif st == "FAILED":
                return status
        await asyncio.sleep(3)
    return {"status": "TIMEOUT"}


async def master_generate(job_input):
    tier = job_input.get("tier", "basic")
    cfg = PLANS.get(tier, PLANS["basic"])
    n = cfg["chunks"]

    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not endpoint_id or not api_key:
        return {"status": "error", "error": "Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY env vars"}

    start = time.time()

    worker_input = {
        "prompt": job_input.get("prompt", ""),
        "negative": job_input.get("negative", ""),
        "style": job_input.get("style", "cinematic"),
        "auto_enhance": job_input.get("auto_enhance", True),
        "width": cfg["width"],
        "height": cfg["height"],
        "length": cfg["length"],
        "steps": cfg["steps"],
        "cfg": cfg["cfg"],
        "sampler": cfg["sampler"],
        "seed": job_input.get("seed", -1),
        "tier": tier,
    }

    async with aiohttp.ClientSession() as session:
        subs = await asyncio.gather(*[submit_chunk(session, endpoint_id, api_key, worker_input, i, n) for i in range(n)])
        results = await asyncio.gather(*[wait_chunk(session, endpoint_id, api_key, s["job_id"], s["chunk"]) for s in subs])

    chunk_files = []
    failed = []
    for i, res in enumerate(results):
        if res.get("status") != "COMPLETED":
            failed.append(i)
            continue
        b64 = res.get("output", {}).get("video_base64")
        if not b64:
            failed.append(i)
            continue
        p = f"/tmp/c_{i:02d}.webp"
        with open(p, "wb") as f:
            f.write(base64.b64decode(b64))
        chunk_files.append(p)

    if not chunk_files:
        return {"status": "error", "error": "All chunks failed", "failed": failed}

    final_path = f"/tmp/final_{int(time.time())}.mp4"
    stitch_chunks(chunk_files, final_path)
    for f in chunk_files:
        os.remove(f)

    with open(final_path, "rb") as f:
        final_b64 = base64.b64encode(f.read()).decode("utf-8")
    sz = round(os.path.getsize(final_path) / (1024*1024), 2)
    os.remove(final_path)
    elapsed = time.time() - start
    final_duration = (len(chunk_files) * CHUNK_DURATION) - ((len(chunk_files) - 1) * CROSSFADE) if len(chunk_files) > 1 else CHUNK_DURATION

    return {
        "status": "completed",
        "tier": tier,
        "plan": "paid",
        "resolution": f"{cfg['width']}x{cfg['height']}",
        "chunks_total": n,
        "chunks_success": len(chunk_files),
        "final_duration_sec": round(final_duration, 1),
        "video_base64": final_b64,
        "size_mb": sz,
        "time_seconds": round(elapsed, 2)
    }


# ═══════════════════════════════════════════════════════════
# BASIC FREE — Direct generate (NO dispatch, saves money!)
# ═══════════════════════════════════════════════════════════
def basic_free_generate(job_input):
    """Basic = FREE. Direct generate on 1 worker, no parallel dispatch."""
    cfg = PLANS["basic"]

    start = time.time()
    wf = build_workflow({
        **job_input,
        "width": cfg["width"],
        "height": cfg["height"],
        "length": cfg["length"],
        "steps": cfg["steps"],
        "cfg": cfg["cfg"],
        "sampler": cfg["sampler"],
    }, 0)

    result = generate_chunk(wf)
    elapsed = time.time() - start

    if result["status"] != "completed":
        return {"status": "error", "error": result.get("error", "Generation failed")}

    return {
        "status": "completed",
        "tier": "basic",
        "plan": "free",
        "resolution": f"{cfg['width']}x{cfg['height']}",
        "final_duration_sec": round(cfg['length'] / 25, 1),
        "video_base64": result["video_base64"],
        "size_mb": result["size_mb"],
        "time_seconds": round(elapsed, 2)
    }


# ═══════════════════════════════════════════════════════════
# MAIN HANDLER
# ═══════════════════════════════════════════════════════════
def handler(job):
    job_input = job["input"]
    tier = job_input.get("tier", "basic")
    cfg = PLANS.get(tier, PLANS["basic"])

    # ═══════════════════════════════════════════════════════
    # FREE TIER: basic — direct generate, NO parallel dispatch
    # Uses only 1 worker, saves money!
    # ═══════════════════════════════════════════════════════
    if tier == "basic" or not cfg["paid"]:
        return basic_free_generate(job_input)

    # ═══════════════════════════════════════════════════════
    # PAID TIERS: standard, pro — parallel chunks + stitch
    # Uses multiple workers, costs more, but FAST!
    # ═══════════════════════════════════════════════════════

    # Worker mode: chunk_index present → generate 1 chunk
    if "chunk_index" in job_input:
        wf = build_workflow({
            **job_input,
            "width": cfg["width"],
            "height": cfg["height"],
            "length": cfg["length"],
            "steps": cfg["steps"],
            "cfg": cfg["cfg"],
            "sampler": cfg["sampler"],
        }, job_input.get("chunk_index", 0))
        start = time.time()
        result = generate_chunk(wf)
        return {
            "status": result["status"],
            "time": round(time.time() - start, 2),
            "tier": tier,
            "chunk_index": job_input.get("chunk_index", 0),
            "total_chunks": job_input.get("total_chunks", 1),
            "video_base64": result.get("video_base64"),
            "filename": result.get("filename"),
            "size_mb": result.get("size_mb"),
            "error": result.get("error")
        }

    # Master mode: dispatch parallel chunks
    return asyncio.run(master_generate(job_input))


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
