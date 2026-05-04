"""
VOC Gems RunPod Serverless Handler
v3: возврат к проверенной 9-апрельской структуре + усиленный негатив с весами
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

comfyui_process = None

def start_comfyui():
    global comfyui_process
    os.chdir("/workspace/ComfyUI")

    lora_src = os.environ.get("LORA_PATH", "/runpod-volume/lora/vocgems_jewelry_v2.safetensors")
    lora_dst = "/workspace/ComfyUI/models/loras/vocgems_jewelry_v2.safetensors"

    if os.path.exists(lora_src) and not os.path.exists(lora_dst):
        os.makedirs("/workspace/ComfyUI/models/loras", exist_ok=True)
        os.symlink(lora_src, lora_dst)
        print(f"LoRA linked: {lora_src} -> {lora_dst}", flush=True)
    elif not os.path.exists(lora_src):
        print(f"WARNING: LoRA not found at {lora_src}", flush=True)

    print("Launching ComfyUI process...", flush=True)
    comfyui_process = subprocess.Popen(
        ["python", "main.py", "--listen", "127.0.0.1", "--port", "8188"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd="/workspace/ComfyUI"
    )

    def log_output():
        for line in comfyui_process.stdout:
            print(f"[ComfyUI] {line.decode('utf-8', errors='ignore').rstrip()}", flush=True)

    threading.Thread(target=log_output, daemon=True).start()
    time.sleep(5)


def wait_for_comfyui():
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


def clean_str(value, default=""):
    if not value:
        return default
    s = str(value).strip()
    if "(" in s:
        s = s.split("(")[0].strip()
    if "," in s:
        s = s.split(",")[0].strip()
    return s or default


def normalize_stone_type(raw):
    s = clean_str(raw, "gemstone").lower()
    mapping = {
        "emeralds": "emerald", "sapphires": "sapphire", "rubies": "ruby",
        "diamonds": "diamond", "spinels": "spinel", "tourmalines": "tourmaline",
        "tanzanites": "tanzanite", "aquamarines": "aquamarine", "topazes": "topaz",
        "garnets": "garnet", "amethysts": "amethyst", "opals": "opal",
        "pearls": "pearl", "morganites": "morganite", "alexandrites": "alexandrite",
    }
    return mapping.get(s, s)


# Короткие якорные фразы — по 9-апрельской системе. Без художественных подсказок.
JEWELRY_ANCHORS = {
    "ring":     "elegant ring",
    "earrings": "drop earrings pair",
    "pendant":  "pendant with delicate chain",
    "necklace": "statement necklace",
    "bracelet": "tennis bracelet",
    "brooch":   "decorative brooch",
}

JEWELRY_NEG = {
    "ring":     "earrings, pendant, necklace, bracelet, brooch",
    "earrings": "ring, pendant, necklace, bracelet, brooch, single earring",
    "pendant":  "ring, earrings, bracelet, brooch",
    "necklace": "ring, earrings, bracelet, brooch",
    "bracelet": "ring, earrings, necklace, pendant, brooch, watch",
    "brooch":   "ring, earrings, necklace, bracelet, pendant",
}

METALS = {
    "gold_750":   "18k yellow gold setting",
    "white_gold": "18k white gold setting",
    "rose_gold":  "18k rose gold setting",
    "platinum":   "platinum setting",
}

STYLES = {
    "classic":   "classic timeless design",
    "modern":    "modern minimalist design",
    "vintage":   "vintage art deco design",
    "statement": "halo setting with diamond accents",
}

STYLE_LEGACY_MAP = {
    "minimalist":  "modern",
    "futuristic":  "modern",
    "geometric":   "modern",
    "artdeco":     "vintage",
    "artnouveau":  "vintage",
    "victorian":   "vintage",
    "halo":        "statement",
    "highjewelry": "statement",
}


def build_prompt(params):
    jewelry_type  = clean_str(params.get("jewelry_type"), "ring").lower()
    stone_type    = normalize_stone_type(params.get("stone_type"))
    stone_color   = clean_str(params.get("stone_color"))
    stone_origin  = clean_str(params.get("stone_origin"))
    stone_cut     = clean_str(params.get("stone_cut")).lower()
    custom_wishes = clean_str(params.get("custom_wishes") or params.get("wishes"))
    metal_key     = clean_str(params.get("metal"), "gold_750").lower()
    style_key     = clean_str(params.get("style"), "modern").lower()
    with_diamonds = bool(params.get("with_diamonds", False))

    try:
        stone_carat = float(params.get("stone_carat") or 1.0)
    except (TypeError, ValueError):
        stone_carat = 1.0

    if style_key in STYLE_LEGACY_MAP:
        style_key = STYLE_LEGACY_MAP[style_key]

    anchor = JEWELRY_ANCHORS.get(jewelry_type, "elegant jewelry")
    weighted_anchor = f"({anchor}:1.4)"

    color_part = f"{stone_color} " if stone_color else ""
    origin_part = f"{stone_origin} " if stone_origin else ""
    cut_part = f"{stone_cut} cut" if stone_cut else "faceted cut"
    stone_desc = f"{stone_carat} carat {color_part}{origin_part}{stone_type}, {cut_part}"

    metal_phrase = METALS.get(metal_key, METALS["gold_750"])
    style_phrase = STYLES.get(style_key, STYLES["modern"])
    diamonds_phrase = ", with small accent diamonds" if with_diamonds else ""
    wishes_phrase = f", {custom_wishes}" if custom_wishes else ""

    # ПОЗИТИВ — короткий, ровно по 9-апрельской структуре
    positive = (
        f"vocgems jewelry, {weighted_anchor}, "
        f"photorealistic jewelry photography, "
        f"{stone_desc}, {metal_phrase}, {style_phrase}{diamonds_phrase}{wishes_phrase}, "
        f"pure white background, studio lighting, "
        f"8k resolution, sharp focus, "
        f"(isolated product shot:1.3), (no people:1.5), product only, "
        f"single piece centered on white seamless background"
    )

    # НЕГАТИВ — усиленный, с весами на анти-человек термины
    type_neg = JEWELRY_NEG.get(jewelry_type, "")
    negative = (
        f"(woman:1.6), (man:1.6), (person:1.6), (human:1.6), (people:1.6), "
        f"(face:1.6), (portrait:1.6), (model:1.6), "
        f"(hand:1.4), (hands:1.4), (fingers:1.4), (skin:1.4), (body:1.4), "
        f"(wearing:1.5), (neck:1.4), (ear:1.4), (wrist:1.4), (arm:1.4), "
        f"earlobe, eye, eyes, hair, lips, mouth, nose, "
        f"mannequin, doll, statue, "
        f"{type_neg}, "
        f"cartoon, illustration, painting, sketch, anime, 3d render, CGI, "
        f"blurry, low quality, deformed, floating stones, "
        f"watermark, text, logo, "
        f"vogue magazine, fashion photography, lifestyle photography, editorial, "
        f"jewelry on model, jewelry being worn"
    )

    print(f"=== PROMPT for {jewelry_type} ({stone_type}) ===", flush=True)
    print(f"POSITIVE: {positive}", flush=True)
    print(f"NEGATIVE: {negative}", flush=True)

    return positive, negative


def get_workflow(positive, negative, seed=None):
    if seed is None:
        seed = int(time.time()) % 1000000000

    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 6,  # снижено с 7 — даёт модели больше свободы следовать промпту
                "denoise": 1,
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


print("Starting ComfyUI...", flush=True)
threading.Thread(target=start_comfyui, daemon=True).start()

runpod.serverless.start({"handler": handler})
