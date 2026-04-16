"""
VOC Gems RunPod Serverless Handler
Генерация ювелирных изображений через ComfyUI
"""

import runpod
import json
import urllib.request
import urllib.parse
import time
import base64
import subprocess
import threading
import os

# Запускаем ComfyUI в фоне
comfyui_process = None

def start_comfyui():
    global comfyui_process
    os.chdir("/workspace/ComfyUI")

    # Линкуем LoRA с Network Volume
    lora_src = os.environ.get("LORA_PATH", "/runpod-volume/lora/vocgems_jewelry_v2.safetensors")
    lora_dst = "/workspace/ComfyUI/models/loras/vocgems_jewelry_v2.safetensors"

    if os.path.exists(lora_src) and not os.path.exists(lora_dst):
        os.makedirs("/workspace/ComfyUI/models/loras", exist_ok=True)
        os.symlink(lora_src, lora_dst)
        print(f"LoRA linked: {lora_src} -> {lora_dst}", flush=True)
    elif not os.path.exists(lora_src):
        print(f"WARNING: LoRA not found at {lora_src}", flush=True)
    else:
        print(f"LoRA already exists at {lora_dst}", flush=True)

    print("Launching ComfyUI process...", flush=True)
    comfyui_process = subprocess.Popen(
        ["python", "main.py", "--listen", "127.0.0.1", "--port", "8188"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd="/workspace/ComfyUI"
    )

    # Читаем логи ComfyUI в отдельном потоке
    def log_output():
        for line in comfyui_process.stdout:
            print(f"[ComfyUI] {line.decode('utf-8', errors='ignore').rstrip()}", flush=True)

    threading.Thread(target=log_output, daemon=True).start()
    time.sleep(5)


def wait_for_comfyui():
    """Ждёт пока ComfyUI запустится — до 3 минут"""
    print("Waiting for ComfyUI to be ready...", flush=True)
    for i in range(180):
        try:
            urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
            print(f"ComfyUI ready after {i} seconds", flush=True)
            return True
        except Exception as e:
            if i % 10 == 0:
                print(f"Still waiting... ({i}s) {e}", flush=True)
            time.sleep(1)
    print("ERROR: ComfyUI did not start in 3 minutes", flush=True)
    return False


def build_prompt(params):
    jewelry_types = {
        "ring": "single elegant ring",
        "earrings": "matching drop earrings pair",
        "pendant": "single pendant necklace with delicate chain",
        "necklace": "single statement necklace"
    }
    metals = {
        "gold_750": "18k yellow gold setting, polished warm gold finish",
        "white_gold": "18k white gold setting, rhodium plated silvery finish",
        "rose_gold": "18k rose gold setting, romantic pink gold tone",
        "platinum": "platinum 950 setting, prestigious cool metal finish"
    }
    styles = {
        "modern": "modern minimalist design, clean lines, contemporary style",
        "classic": "classic timeless design, traditional elegant setting",
        "artdeco": "art deco geometric design, 1920s inspired, symmetric patterns",
        "halo": "halo setting surrounded by brilliant diamonds, pave accents"
    }

    jewelry = jewelry_types.get(params.get("jewelry_type", "ring"), "elegant jewelry")
    metal = metals.get(params.get("metal", "gold_750"), "18k gold setting")
    style = styles.get(params.get("style", "modern"), "elegant design")
    stone_type = params.get("stone_type", "emerald")
    stone_carat = params.get("stone_carat", 3.0)
    stone_color = params.get("stone_color", "vivid green")
    stone_origin = params.get("stone_origin", "")
    stone_cut = params.get("stone_cut", "emerald cut")
    with_diamonds = params.get("with_diamonds", False)
    custom_wishes = params.get("custom_wishes", "")

    origin = f" {stone_origin}" if stone_origin else ""
    diamonds = ", with small accent diamonds" if with_diamonds else ""
    wishes = f", {custom_wishes}" if custom_wishes else ""

    positive = (
        f"vocgems jewelry, photorealistic jewelry product photography, studio lighting, "
        f"{jewelry} with {stone_carat} carat {stone_color}{origin} {stone_type}, "
        f"{stone_cut} cut, natural gemstone, excellent clarity, "
        f"{metal}, {style}{diamonds}{wishes}, "
        f"pure white background, soft professional lighting, subtle shadows, "
        f"8k resolution, highly detailed, sharp focus, crystal clear gemstone facets, "
        f"brilliant reflections, luxury jewelry catalogue, commercial advertising quality, "
        f"isolated product shot, single item centered, no people, product only, one piece only"
    )
    negative = (
        "woman, man, person, human, people, hand, hands, fingers, face, body, skin, "
        "portrait, model, wearing, neck, ear, wrist, arm, nude, nsfw, "
        "multiple items, two rings, pair of rings, many pieces, duplicates, copies, "
        "cartoon, illustration, painting, sketch, 3d render, CGI, anime, fantasy, unrealistic, "
        "blurry, low quality, pixelated, watermark, text, logo, "
        "bad proportions, deformed, distorted metal, floating stones, impossible geometry"
    )
    return positive, negative


def get_workflow(positive, negative, seed=None):
    if seed is None:
        seed = int(time.time()) % 1000000000

    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 7, "denoise": 1,
                "latent_image": ["5", 0], "model": ["10", 0],
                "negative": ["7", 0], "positive": ["6", 0],
                "sampler_name": "dpmpp_2m", "scheduler": "karras",
                "seed": seed, "steps": 30
            }
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "epicRealism.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": 768, "width": 768}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["10", 1], "text": positive}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["10", 1], "text": negative}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "vocgems", "images": ["8", 0]}},
        "10": {
            "class_type": "LoraLoader",
            "inputs": {
                "clip": ["4", 1], "lora_name": "vocgems_jewelry_v2.safetensors",
                "model": ["4", 0], "strength_clip": 0.5, "strength_model": 0.5
            }
        }
    }


def queue_prompt(workflow):
    data = json.dumps({"prompt": workflow}).encode('utf-8')
    req = urllib.request.Request(
        "http://127.0.0.1:8188/prompt",
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))


def get_image(filename):
    url = f"http://127.0.0.1:8188/view?filename={filename}&type=output"
    with urllib.request.urlopen(url) as response:
        return base64.b64encode(response.read()).decode('utf-8')


def wait_for_completion(prompt_id, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            url = f"http://127.0.0.1:8188/history/{prompt_id}"
            with urllib.request.urlopen(url) as response:
                history = json.loads(response.read().decode('utf-8'))
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    for node_id, output in outputs.items():
                        if "images" in output:
                            return output["images"][0]["filename"]
        except:
            pass
        time.sleep(1)
    return None


def handler(job):
    job_input = job.get("input", {})
    print(f"Job received: {job_input}", flush=True)

    if not wait_for_comfyui():
        return {"error": "ComfyUI failed to start"}

    positive, negative = build_prompt(job_input)
    workflow = get_workflow(positive, negative)

    try:
        result = queue_prompt(workflow)
    except Exception as e:
        return {"error": f"Failed to queue prompt: {str(e)}"}

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        return {"error": "Failed to queue prompt"}

    print(f"Prompt queued: {prompt_id}", flush=True)
    filename = wait_for_completion(prompt_id)

    if not filename:
        return {"error": "Generation timeout"}

    print(f"Generation complete: {filename}", flush=True)
    image_base64 = get_image(filename)

    return {"image": image_base64, "prompt_id": prompt_id, "filename": filename}


# Запускаем ComfyUI при старте
print("Starting ComfyUI...", flush=True)
threading.Thread(target=start_comfyui, daemon=True).start()

runpod.serverless.start({"handler": handler})
