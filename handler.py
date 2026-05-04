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


# ─── НОРМАЛИЗАЦИЯ ВХОДНЫХ ДАННЫХ ──────────────────────────────────────────────

def clean_str(value, default=""):
    """Чистит строку от пробелов, пометок в скобках, мусора после запятой."""
    if not value:
        return default
    s = str(value).strip()
    # Убираем всё в скобках: "Vivid Green (GIA)" -> "Vivid Green"
    if "(" in s:
        s = s.split("(")[0].strip()
    # Убираем всё после запятой: "Royal Blue, no heat" -> "Royal Blue"
    if "," in s:
        s = s.split(",")[0].strip()
    return s or default


def normalize_stone_type(raw):
    """Приводит stone_type из WDK к каноническому виду в единственном числе."""
    s = clean_str(raw, "gemstone").lower()
    # Маппинг множественного числа и вариаций
    mapping = {
        "emeralds": "emerald", "emerald": "emerald",
        "sapphires": "sapphire", "sapphire": "sapphire",
        "rubies": "ruby", "ruby": "ruby",
        "diamonds": "diamond", "diamond": "diamond",
        "spinels": "spinel", "spinel": "spinel",
        "tourmalines": "tourmaline", "tourmaline": "tourmaline",
        "tanzanites": "tanzanite", "tanzanite": "tanzanite",
        "aquamarines": "aquamarine", "aquamarine": "aquamarine",
        "topazes": "topaz", "topaz": "topaz",
        "garnets": "garnet", "garnet": "garnet",
        "amethysts": "amethyst", "amethyst": "amethyst",
        "opals": "opal", "opal": "opal",
        "pearls": "pearl", "pearl": "pearl",
        "morganites": "morganite", "morganite": "morganite",
        "alexandrites": "alexandrite", "alexandrite": "alexandrite",
    }
    return mapping.get(s, s)  # неизвестные типы пропускаем как есть


# ─── ШАБЛОНЫ ПО ТИПУ ИЗДЕЛИЯ ──────────────────────────────────────────────────
#
# Принцип: для каждого типа — свой шаблон композиции и свой негатив, который
# жёстко запрещает все остальные типы. Тип изделия идёт ПЕРВЫМ словом в промпте,
# повторяется 3+ раза (в начале, в середине, в конце) для закрепления концепта.
#
# Каждый шаблон — функция, которая получает (stone_phrase, metal, style_phrase,
# diamonds_phrase, wishes_phrase) и возвращает (positive, negative).

# Универсальная "хвостовая" часть позитива — техническое качество фото
QUALITY_TAIL = (
    "pure white seamless background, soft professional studio lighting, subtle shadows, "
    "8k resolution, ultra detailed, sharp focus, crystal clear gemstone facets, "
    "brilliant light reflections, luxury jewelry catalogue photography, "
    "commercial advertising quality, macro photography"
)

# Универсальная "хвостовая" часть негатива — общее качество и анти-человек
QUALITY_NEG = (
    "woman, man, person, human, people, hand, hands, fingers, face, body, skin, "
    "portrait, model, wearing, neck, ear, wrist, arm, finger, mannequin, nude, nsfw, "
    "cartoon, illustration, painting, sketch, drawing, anime, fantasy, unrealistic, "
    "3d render, CGI, plastic, toy, "
    "blurry, out of focus, low quality, pixelated, watermark, text, logo, signature, "
    "bad proportions, deformed, distorted metal, floating stones, impossible geometry, "
    "cluttered background, busy background, multiple backgrounds"
)


def build_ring_prompt(stone, metal, style, diamonds, wishes):
    positive = (
        f"a single ring, one ring, ring jewelry, "
        f"vocgems jewelry, photorealistic product photography of one ring, "
        f"the ring features {stone}, {stone} set in {metal}, "
        f"{style}{diamonds}{wishes}, "
        f"single ring centered in frame, top-down three-quarter view of one ring, "
        f"{QUALITY_TAIL}"
    )
    negative = (
        "earrings, pair of earrings, two earrings, drop earrings, stud earrings, "
        "pendant, necklace, chain, bracelet, brooch, pin, "
        "two rings, multiple rings, pair of rings, "
        "human ear, human finger, "
        + QUALITY_NEG
    )
    return positive, negative


def build_earrings_prompt(stone, metal, style, diamonds, wishes):
    positive = (
        f"a pair of earrings, two matching earrings, earring pair, "
        f"vocgems jewelry, photorealistic product photography of a pair of earrings, "
        f"two identical earrings displayed side by side, both earrings visible, "
        f"each earring features {stone}, {stone} set in {metal}, "
        f"{style}{diamonds}{wishes}, "
        f"symmetrical pair of earrings centered in frame, "
        f"matching pair of earrings, mirror image earrings, "
        f"{QUALITY_TAIL}"
    )
    negative = (
        "ring, single ring, one ring, finger ring, "
        "pendant, necklace, chain, bracelet, brooch, pin, "
        "single earring, just one earring, lone earring, mismatched earrings, "
        "asymmetric earrings, three earrings, "
        + QUALITY_NEG
    )
    return positive, negative


def build_pendant_prompt(stone, metal, style, diamonds, wishes):
    positive = (
        f"a pendant on a chain, pendant necklace, single pendant, "
        f"vocgems jewelry, photorealistic product photography of a pendant, "
        f"the pendant features {stone}, {stone} set in {metal} pendant setting, "
        f"{style}{diamonds}{wishes}, "
        f"delicate {metal} chain, pendant hanging from thin chain, "
        f"single pendant centered in frame, vertical orientation, "
        f"{QUALITY_TAIL}"
    )
    negative = (
        "ring, finger ring, "
        "earrings, pair of earrings, two earrings, "
        "bracelet, wrist jewelry, brooch, pin, "
        "thick chain, choker, multiple pendants, "
        + QUALITY_NEG
    )
    return positive, negative


def build_necklace_prompt(stone, metal, style, diamonds, wishes):
    positive = (
        f"a statement necklace, full necklace, fine jewelry necklace, "
        f"vocgems jewelry, photorealistic product photography of a necklace, "
        f"the necklace features {stone} as centerpiece, {stone} set in {metal}, "
        f"{style}{diamonds}{wishes}, "
        f"complete necklace laid flat in U-shape, full chain visible, "
        f"single necklace centered in frame, "
        f"{QUALITY_TAIL}"
    )
    negative = (
        "ring, finger ring, "
        "earrings, pair of earrings, two earrings, "
        "bracelet, wrist jewelry, brooch, pin, "
        "small pendant only, just a pendant, partial necklace, "
        + QUALITY_NEG
    )
    return positive, negative


def build_bracelet_prompt(stone, metal, style, diamonds, wishes):
    positive = (
        f"a bracelet, fine jewelry bracelet, single bracelet, "
        f"vocgems jewelry, photorealistic product photography of a bracelet, "
        f"the bracelet features {stone}, {stone} set in {metal} bracelet, "
        f"{style}{diamonds}{wishes}, "
        f"complete bracelet laid flat in circular shape, full bracelet visible, "
        f"single bracelet centered in frame, horizontal orientation, "
        f"{QUALITY_TAIL}"
    )
    negative = (
        "ring, finger ring, "
        "earrings, pair of earrings, two earrings, "
        "necklace, chain, pendant, brooch, pin, "
        "watch, wristwatch, multiple bracelets, stacked bracelets, "
        + QUALITY_NEG
    )
    return positive, negative


def build_brooch_prompt(stone, metal, style, diamonds, wishes):
    positive = (
        f"a brooch, decorative brooch pin, single brooch, "
        f"vocgems jewelry, photorealistic product photography of a brooch, "
        f"the brooch features {stone}, {stone} set in {metal} brooch, "
        f"ornamental brooch with pin back, {style}{diamonds}{wishes}, "
        f"single brooch centered in frame, top-down view of brooch, "
        f"{QUALITY_TAIL}"
    )
    negative = (
        "ring, finger ring, "
        "earrings, pair of earrings, two earrings, "
        "necklace, chain, pendant, bracelet, "
        "fabric, clothing, garment, "
        "multiple brooches, "
        + QUALITY_NEG
    )
    return positive, negative


# Роутер — выбирает шаблон по типу
JEWELRY_BUILDERS = {
    "ring":     build_ring_prompt,
    "earrings": build_earrings_prompt,
    "pendant":  build_pendant_prompt,
    "necklace": build_necklace_prompt,
    "bracelet": build_bracelet_prompt,
    "brooch":   build_brooch_prompt,
}


# ─── СЛОВАРИ ДЛЯ МЕТАЛЛА И СТИЛЯ ──────────────────────────────────────────────

METALS = {
    "gold_750":   "18k yellow gold, polished warm gold finish",
    "white_gold": "18k white gold, rhodium plated silvery finish",
    "rose_gold":  "18k rose gold, romantic pink gold tone",
    "platinum":   "platinum 950, prestigious cool metal finish",
}

# Сокращённый набор стилей — 4 вместо 10. Меньше противоречий в промпте.
# Если фронт пришлёт старое значение — мапим в одно из этих 4.
STYLES = {
    "classic":   "classic timeless elegant design, refined traditional setting",
    "modern":    "modern minimalist design, clean geometric lines, contemporary",
    "vintage":   "vintage inspired design, art deco details, ornate antique style",
    "statement": "bold statement design, halo setting with pave diamond accents, dramatic",
}

# Маппинг старых значений стиля в новые 4
STYLE_LEGACY_MAP = {
    "minimalist":   "modern",
    "futuristic":   "modern",
    "artdeco":      "vintage",
    "artnouveau":   "vintage",
    "victorian":    "vintage",
    "geometric":    "modern",
    "halo":         "statement",
    "highjewelry":  "statement",
}


def build_prompt(params):
    # 1. Чистим и нормализуем все входные параметры
    jewelry_type  = clean_str(params.get("jewelry_type"), "ring").lower()
    stone_type    = normalize_stone_type(params.get("stone_type"))
    stone_color   = clean_str(params.get("stone_color"))
    stone_origin  = clean_str(params.get("stone_origin"))
    stone_cut     = clean_str(params.get("stone_cut")).lower()
    custom_wishes = clean_str(params.get("custom_wishes") or params.get("wishes"))
    metal_key     = clean_str(params.get("metal"), "gold_750").lower()
    style_key     = clean_str(params.get("style"), "modern").lower()
    with_diamonds = bool(params.get("with_diamonds", False))

    # stone_carat — число
    try:
        stone_carat = float(params.get("stone_carat") or 1.0)
    except (TypeError, ValueError):
        stone_carat = 1.0

    # 2. Маппим стиль (старые значения -> новые 4)
    if style_key in STYLE_LEGACY_MAP:
        style_key = STYLE_LEGACY_MAP[style_key]

    # 3. Собираем "фразу о камне" — она вставляется в позитив несколько раз
    color_part = f"{stone_color} " if stone_color else ""
    origin_part = f"from {stone_origin}" if stone_origin else ""
    cut_part = f"{stone_cut} cut" if stone_cut else "faceted cut"

    stone_phrase = (
        f"a {stone_carat} carat {color_part}natural {stone_type} {origin_part}, "
        f"{cut_part}, excellent clarity, brilliant gemstone"
    ).strip()

    # 4. Подставляем металл и стиль
    metal_phrase = METALS.get(metal_key, METALS["gold_750"])
    style_phrase = STYLES.get(style_key, STYLES["modern"])

    diamonds_phrase = ", with small accent diamonds" if with_diamonds else ""
    wishes_phrase = f", {custom_wishes}" if custom_wishes else ""

    # 5. Выбираем шаблон по типу изделия
    builder = JEWELRY_BUILDERS.get(jewelry_type)
    if not builder:
        # Неизвестный тип — фоллбэк на ring, чтобы хоть что-то осмысленное
        print(f"WARNING: unknown jewelry_type '{jewelry_type}', falling back to ring", flush=True)
        builder = build_ring_prompt

    positive, negative = builder(
        stone_phrase, metal_phrase, style_phrase, diamonds_phrase, wishes_phrase
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
