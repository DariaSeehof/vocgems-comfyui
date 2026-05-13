"""
VOC Gems RunPod Serverless Handler
v6.6: поддержка бледных/прозрачных камней (Moonstone, Opal, Pearl, Diamond,
      Pastel/Light/Milky-цвета).
      - WEAK_VISUAL_STONES: камни с слабым цветовым сигналом → Tile понижен
      - PALE_COLOR_MARKERS: бледные цвета (pastel, light, milky) → Tile средний
      - STONE_DESCRIPTORS: специфичные якоря по типу камня
      - CABOCHON_DEFAULT_STONES: лунный/опал/жемчуг идут как cabochon
      - Anti-warm-tone защита для прозрачных камней (мягче, вес 1.4)
      - Adaptive ControlNet: strength/end_percent зависят от силы сигнала
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

# ─── Базовая модель (checkpoint) ───
CHECKPOINT_NAME = "juggernaut_reborn.safetensors"

# ─── ControlNet параметры ───
# Tile передаёт ЦВЕТ и общую структуру референса, а не контуры.
# v6.6: три пресета силы. Strong для ярких/чётких камней, weak для прозрачных
# (чтобы Tile не тащил белый фон референса в композицию).
CONTROLNET_MODEL = "control_v11f1e_sd15_tile.pth"

CONTROLNET_PRESETS = {
    "strong": {"strength": 0.4, "start": 0.0, "end": 0.5},   # яркие камни (рубеллит, изумруд)
    "medium": {"strength": 0.3, "start": 0.0, "end": 0.4},   # бледные цветные (Pastel Pink спинель)
    "weak":   {"strength": 0.2, "start": 0.0, "end": 0.3},   # прозрачные (лунный камень, опал, бриллиант)
}
CONTROLNET_DEFAULT_PRESET = "strong"
REFERENCE_FILENAME = "vocgems_reference.png"


def start_comfyui():
    global comfyui_process
    os.chdir("/workspace/ComfyUI")

    # ─── Симлинк LoRA ───
    lora_src = os.environ.get("LORA_PATH", "/runpod-volume/lora/vocgems_jewelry_v2.safetensors")
    lora_dst = "/workspace/ComfyUI/models/loras/vocgems_jewelry_v2.safetensors"

    if os.path.exists(lora_src) and not os.path.exists(lora_dst):
        os.makedirs("/workspace/ComfyUI/models/loras", exist_ok=True)
        os.symlink(lora_src, lora_dst)
        print(f"LoRA linked: {lora_src} -> {lora_dst}", flush=True)
    elif not os.path.exists(lora_src):
        print(f"WARNING: LoRA not found at {lora_src}", flush=True)

    # ─── Симлинк Checkpoint ───
    ckpt_src = os.environ.get("CKPT_PATH", f"/runpod-volume/checkpoints/{CHECKPOINT_NAME}")
    ckpt_dst = f"/workspace/ComfyUI/models/checkpoints/{CHECKPOINT_NAME}"

    if os.path.exists(ckpt_src) and not os.path.exists(ckpt_dst):
        os.makedirs("/workspace/ComfyUI/models/checkpoints", exist_ok=True)
        os.symlink(ckpt_src, ckpt_dst)
        print(f"Checkpoint linked: {ckpt_src} -> {ckpt_dst}", flush=True)
    elif not os.path.exists(ckpt_src):
        print(f"WARNING: Checkpoint not found at {ckpt_src}", flush=True)

    # ─── Симлинк ControlNet ───
    cnet_src = os.environ.get("CONTROLNET_PATH", f"/runpod-volume/controlnet/{CONTROLNET_MODEL}")
    cnet_dst = f"/workspace/ComfyUI/models/controlnet/{CONTROLNET_MODEL}"

    if os.path.exists(cnet_src) and not os.path.exists(cnet_dst):
        os.makedirs("/workspace/ComfyUI/models/controlnet", exist_ok=True)
        os.symlink(cnet_src, cnet_dst)
        print(f"ControlNet linked: {cnet_src} -> {cnet_dst}", flush=True)
    elif not os.path.exists(cnet_src):
        print(f"WARNING: ControlNet not found at {cnet_src} — генерация будет без ControlNet", flush=True)

    # ─── Папка input для reference картинок ───
    os.makedirs("/workspace/ComfyUI/input", exist_ok=True)

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


# ─── ХЕЛПЕРЫ ДАННЫХ ───────────────────────────────────────────────────────────

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
    # v6.5: ослаблено с 1.5+1.4+1.3 (три фразы) до одной фразы с весом 1.3
    # Металл не должен перебивать камень — баланс восстановлен в пользу цвета камня (1.5-1.7)
    "gold_750":   "(18k yellow gold band:1.3), warm gold setting",
    "white_gold": "(18k white gold band:1.3), polished silver-tone setting",
    "rose_gold":  "(18k rose gold band:1.3), warm pink gold setting",
    "platinum":   "(platinum band:1.3), polished cool-tone setting",
}

# Тон металла: warm = тянет картинку в жёлтый/оранжевый, cool = в серебристый
# Используется для защиты камня в негативе
METAL_TONE = {
    "gold_750":   "warm",
    "rose_gold":  "warm",
    "white_gold": "cool",
    "platinum":   "cool",
}

# Анти-металл в негатив — чтобы Tile не перебивал цветовую гамму выбранным золотом
METAL_NEG = {
    "gold_750":   "(silver metal:1.5), (white gold:1.5), (platinum:1.4), (rose gold:1.4), cool tone metal",
    "white_gold": "(yellow gold:1.4), (rose gold:1.4), warm tone metal, golden tint",
    "rose_gold":  "(yellow gold:1.4), (white gold:1.4), (silver:1.4), (platinum:1.4), cool tone metal",
    "platinum":   "(yellow gold:1.4), (rose gold:1.4), warm tone metal, golden tint",
}

STYLES = {
    "classic":    "(classic timeless design:1.3), elegant traditional style",
    "modern":     "(modern minimalist design:1.3), clean lines, contemporary",
    "vintage":    "(vintage art deco design:1.3), geometric patterns, 1920s aesthetic",
    "halo":       "(halo setting:1.4), diamond halo around center stone, pavé accents",
    "solitaire":  "(solitaire ring:1.4), single center stone, classic prong setting, minimalist",
    "statement":  "(bold statement design:1.4), large dramatic piece, high jewelry style",
    "nature":     "(nature inspired design:1.3), organic flowing forms, leaf and vine motifs",
    "geometric":  "(geometric design:1.3), sharp angular forms, architectural lines",
}

STYLE_LEGACY_MAP = {
    # старые ключи если придут со старого фронта
    "minimalist":  "modern",
    "futuristic":  "modern",
    "artdeco":     "vintage",
    "artnouveau":  "nature",
    "victorian":   "vintage",
    "highjewelry": "statement",
    "pave":        "halo",
    "floral":      "nature",
}


# ─── ЦВЕТОВАЯ ЛОГИКА ──────────────────────────────────────────────────────────

STONE_DEFAULT_COLORS = {
    "ruby":        ["red"],
    "emerald":     ["green"],
    "sapphire":    ["blue", "deep blue", "royal blue"],
    "spinel":      ["pink", "red", "hot pink"],
    "tourmaline":  ["pink", "green"],
    "tanzanite":   ["violet", "purple-blue"],
    "aquamarine":  ["light blue"],
    "topaz":       ["yellow", "blue"],
    "garnet":      ["red", "deep red"],
    "amethyst":    ["purple"],
    "morganite":   ["pink", "peach"],
    "alexandrite": ["green"],
    "diamond":     ["white", "colorless"],
    "pearl":       ["white"],
    "opal":        ["white"],
}

UNUSUAL_COLOR_MARKERS = {
    # Интенсивность (v6.5: добавлено — чтобы "Vivid Pink" триггерил защиту цвета)
    "vivid", "bright", "intense", "hot", "saturated", "electric", "neon", "rich",
    # Светлота / приглушённость
    "pastel", "light", "pale", "soft", "muted",
    "dark", "deep",
    # Оттенки (расширено v6.5)
    "grey", "gray", "champagne", "peach", "salmon",
    "cognac", "honey", "lavender", "lilac", "mint",
    "teal", "olive", "smoky", "smokey",
    "crimson", "raspberry", "magenta", "fuchsia", "rubellite", "wine",
    "burgundy", "coral", "apricot", "amber",
}

# v6.6: камни со слабым цветовым сигналом на фото (прозрачные, молочные, перламутровые).
# Для них Tile НЕ должен тащить контекст референса — иначе камень исчезает,
# а в кадр приходит белая замша/палец/коробка с реф-фото.
WEAK_VISUAL_STONES = {
    "moonstone", "opal", "pearl", "diamond",
}

# v6.6: маркеры бледных цветов — для них Tile среднего уровня
# (камень есть, но цветовой сигнал на референсе слабый)
PALE_COLOR_MARKERS = {
    "pastel", "light", "pale", "soft", "milky", "dusty", "powder", "baby",
}

# v6.6: специфичные дескрипторы по типу камня. Используются как якорь в позитиве
# ВМЕСТО универсального `vivid {color} {stone} center stone`. Для камней которых
# нет в этом словаре — fallback на универсальный якорь.
STONE_DESCRIPTORS = {
    "moonstone":  "(adularescent moonstone:1.5), (milky white cabochon with blue sheen:1.4), translucent gem",
    "opal":       "(opalescent gemstone:1.5), (rainbow play of color:1.4), (iridescent:1.3)",
    "pearl":      "(lustrous pearl:1.5), (iridescent pearl surface:1.4), nacre shimmer",
    "diamond":    "(brilliant cut diamond:1.5), (fire and scintillation:1.4), colorless transparent gemstone",
    "tanzanite":  "(tanzanite gemstone:1.5), (violet-blue dichroic:1.4)",
    "alexandrite":"(alexandrite gemstone:1.5), (color-changing green-purple:1.4)",
}

# v6.6: камни которые в реальном каталоге почти всегда кабошоны.
# Если фронт прислал stone_cut типа "Cushion" для лунного камня — игнорируем,
# ставим cabochon. Сохраняем "огранка" как opt-out: если в stone_cut есть слова
# "faceted/round/oval/cushion/emerald/marquise" И это лунный камень — оставляем
# как есть (значит редкий фасетированный экземпляр).
CABOCHON_DEFAULT_STONES = {
    "moonstone", "opal", "pearl",
}


def get_controlnet_preset(stone_type, stone_color):
    """Выбирает пресет силы ControlNet по типу камня и цвету.
    weak — прозрачные/молочные камни (moonstone, opal, pearl, diamond)
    medium — камни с бледным цветом (Pastel Pink spinel)
    strong — всё остальное (яркие цветные камни)"""
    if stone_type in WEAK_VISUAL_STONES:
        return "weak"
    if stone_color:
        color_lower = stone_color.lower()
        for marker in PALE_COLOR_MARKERS:
            if marker in color_lower:
                return "medium"
    return "strong"


def build_stone_descriptor(stone_type, stone_color):
    """Возвращает якорь-описание камня для позитива (вставляется после металла).
    Специфичный по типу если есть в STONE_DESCRIPTORS, иначе универсальный по цвету."""
    if stone_type in STONE_DESCRIPTORS:
        return STONE_DESCRIPTORS[stone_type]
    if stone_color:
        return f"(vivid {stone_color} {stone_type} center stone:1.4)"
    return ""


def is_unusual_color(stone_color, stone_type):
    if not stone_color:
        return False
    color_lower = stone_color.lower()
    for marker in UNUSUAL_COLOR_MARKERS:
        if marker in color_lower:
            return True
    defaults = STONE_DEFAULT_COLORS.get(stone_type, [])
    if defaults:
        for default in defaults:
            if default in color_lower:
                return False
        return True
    return False


def build_color_negative(stone_color, stone_type):
    if not is_unusual_color(stone_color, stone_type):
        return ""
    defaults = STONE_DEFAULT_COLORS.get(stone_type, [])
    if not defaults:
        return ""
    color_lower = stone_color.lower() if stone_color else ""
    filtered = [d for d in defaults if d not in color_lower and color_lower not in d]
    if not filtered:
        return ""
    return ", ".join(f"{d} {stone_type}" for d in filtered)


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

    unusual = is_unusual_color(stone_color, stone_type)
    color_weight = 1.7 if unusual else 1.5

    if stone_color:
        weighted_color = f"({stone_color}:{color_weight})"
        color_emphasis = f"{weighted_color} {stone_type}, {stone_color} colored gemstone, "
    else:
        color_emphasis = f"{stone_type}, "

    # v6.6: для лунного камня/опала/жемчуга принудительно cabochon
    # (фронт может прислать "Cushion" из WDK — но это почти всегда ошибка
    # для этих типов; реальный каталог = выпуклый кабошон).
    if stone_type in CABOCHON_DEFAULT_STONES:
        cut_part = "(cabochon:1.4)"
    elif stone_cut:
        cut_part = f"{stone_cut} cut"
    else:
        cut_part = "faceted cut"

    origin_part = f", from {stone_origin}" if stone_origin else ""
    stone_desc = f"{color_emphasis}{stone_carat} carat, {cut_part}{origin_part}"

    metal_phrase = METALS.get(metal_key, METALS["gold_750"])
    style_phrase = STYLES.get(style_key, STYLES["modern"])
    diamonds_phrase = ", with small accent diamonds" if with_diamonds else ""
    wishes_phrase = f", {custom_wishes}" if custom_wishes else ""

    # v6.6: якорь — специфичный по типу камня если есть в STONE_DESCRIPTORS,
    # иначе универсальный по цвету (как в v6.5)
    stone_descriptor = build_stone_descriptor(stone_type, stone_color)
    descriptor_part = f", {stone_descriptor}" if stone_descriptor else ""

    # v6.6: выбор пресета силы ControlNet
    cn_preset_name = get_controlnet_preset(stone_type, stone_color)
    cn_preset = CONTROLNET_PRESETS[cn_preset_name]

    # v6.5: новый порядок — камень → стиль/композиция → металл → ЯКОРЬ цвета камня.
    # Это даёт модели сначала "увидеть" камень, потом стиль, и только в конце уточнить металл.
    # Якорь цвета после металла блокирует перетекание цвета с золота на камень.
    positive = (
        f"vocgems jewelry, {weighted_anchor}, "
        f"(professional product catalog photography:1.5), "
        f"(commercial jewelry catalog:1.4), "
        f"{stone_desc}, "
        f"{style_phrase}{diamonds_phrase}{wishes_phrase}, "
        f"{metal_phrase}"
        f"{descriptor_part}, "
        f"(pure white seamless background:1.6), "
        f"(plain white studio backdrop:1.5), "
        f"professional studio lighting, soft shadows, "
        f"8k resolution, sharp focus, "
        f"(isolated product shot:1.4), (no people:1.6), product only, "
        f"single piece centered, jewelry store catalog aesthetic"
    )

    type_neg = JEWELRY_NEG.get(jewelry_type, "")
    color_neg = build_color_negative(stone_color, stone_type)
    color_neg_part = f"{color_neg}, " if color_neg else ""
    metal_neg = METAL_NEG.get(metal_key, "")
    metal_neg_part = f"{metal_neg}, " if metal_neg else ""

    # v6.5/v6.6: защита камня от тёплого металла.
    # v6.5: только cold-coloured камни при warm-металле, вес 1.6
    # v6.6: + прозрачные камни (weak_visual) при warm-металле, вес 1.4 (мягче,
    # чтобы не сделать камень совсем бесцветным и не убрать золотые рефлексы)
    metal_tone = METAL_TONE.get(metal_key, "warm")
    cold_stone_colors = {"pink", "blue", "green", "violet", "purple", "magenta",
                         "fuchsia", "lavender", "lilac", "teal", "mint", "raspberry",
                         "crimson", "rubellite"}
    stone_is_cold = stone_color and any(c in stone_color.lower() for c in cold_stone_colors)
    stone_is_transparent = stone_type in WEAK_VISUAL_STONES

    stone_color_anti_metal = ""
    if metal_tone == "warm" and stone_is_cold:
        stone_color_anti_metal = (
            f"(yellow {stone_type}:1.6), (gold colored stone:1.5), "
            f"(yellow gemstone:1.5), (warm tinted stone:1.4), "
            f"(metallic colored stone:1.4), "
        )
    elif metal_tone == "warm" and stone_is_transparent:
        # Прозрачный камень не должен пожелтеть от рефлексов золота, но не давим
        # слишком сильно — рефлексы должны остаться
        stone_color_anti_metal = (
            f"(yellow {stone_type}:1.4), (gold colored {stone_type}:1.3), "
            f"(opaque yellow stone:1.3), "
        )

    negative = (
        f"(woman:1.6), (man:1.6), (person:1.6), (human:1.6), (people:1.6), "
        f"(face:1.6), (portrait:1.6), (model:1.6), "
        f"(hand:1.7), (hands:1.7), (fingers:1.7), (holding:1.6), "
        f"(skin:1.5), (body:1.4), (palm:1.5), (fingernail:1.5), (nail:1.4), "
        f"(wearing:1.5), (neck:1.4), (ear:1.4), (wrist:1.4), (arm:1.4), "
        f"earlobe, eye, eyes, hair, lips, mouth, nose, "
        f"mannequin, doll, statue, "
        f"{type_neg}, "
        f"{color_neg_part}"
        f"{stone_color_anti_metal}"
        f"{metal_neg_part}"
        f"(flower:1.5), (petals:1.5), (leaves:1.4), (plants:1.4), "
        f"(fabric:1.5), (cloth:1.5), (silk:1.4), (paper:1.4), (textured background:1.4), "
        f"(colored background:1.5), (pastel background:1.5), (artistic background:1.5), "
        f"(creative composition:1.4), (lifestyle setting:1.4), "
        f"cartoon, illustration, painting, sketch, anime, 3d render, CGI, "
        f"blurry, soft focus, low quality, deformed, floating stones, "
        f"watermark, text, logo, "
        f"vogue magazine, fashion photography, lifestyle photography, editorial, "
        f"jewelry on model, jewelry being worn, jewelry being held, "
        f"two stones, multiple gems, pearl, pearls, sphere, beads"
    )

    # v6.6: лунный/опал/жемчуг — НЕ должны попадать в "pearl, pearls, sphere, beads"
    # в негативе, иначе модель будет избегать сферической/перламутровой формы
    if stone_type in {"moonstone", "opal", "pearl"}:
        negative = negative.replace(", pearl, pearls, sphere, beads", "")

    print(f"=== PROMPT for {jewelry_type} ({stone_type}) ===", flush=True)
    print(f"=== unusual_color={unusual}, metal_tone={metal_tone}, "
          f"stone_is_cold={stone_is_cold}, stone_is_transparent={stone_is_transparent}, "
          f"cn_preset={cn_preset_name} (strength={cn_preset['strength']}, end={cn_preset['end']}) ===",
          flush=True)
    print(f"POSITIVE: {positive}", flush=True)
    print(f"NEGATIVE: {negative}", flush=True)

    return positive, negative, cn_preset


# ─── REFERENCE IMAGE DOWNLOAD ─────────────────────────────────────────────────

def download_reference_image(url, timeout=15):
    """Скачивает фото камня по URL в /workspace/ComfyUI/input/.
    Возвращает имя файла (без пути) или None при ошибке."""
    if not url:
        return None

    try:
        print(f"Downloading reference image: {url}", flush=True)
        req = urllib.request.Request(url, headers={
            'User-Agent': 'VOCGems-RunPod-Worker/1.0'
        })
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
            content_type = response.headers.get('Content-Type', '').lower()

        if not data or len(data) < 100:
            print(f"WARNING: reference image too small ({len(data)} bytes)", flush=True)
            return None

        # Определяем расширение
        if 'png' in content_type:
            ext = 'png'
        elif 'webp' in content_type:
            ext = 'webp'
        else:
            ext = 'jpg'

        filename = f"vocgems_reference.{ext}"
        path = f"/workspace/ComfyUI/input/{filename}"
        with open(path, 'wb') as f:
            f.write(data)

        print(f"Reference image saved: {path} ({len(data)} bytes, {content_type})", flush=True)
        return filename
    except Exception as e:
        print(f"WARNING: failed to download reference image: {e}", flush=True)
        return None


# ─── WORKFLOWS ────────────────────────────────────────────────────────────────

def get_workflow_basic(positive, negative, seed=None):
    """Старый workflow без ControlNet — fallback."""
    if seed is None:
        seed = int(time.time()) % 1000000000

    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 6,
                "denoise": 1,
                "latent_image": ["5", 0], "model": ["10", 0],
                "negative": ["7", 0], "positive": ["6", 0],
                "sampler_name": "dpmpp_2m", "scheduler": "karras",
                "seed": seed, "steps": 30
            }
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT_NAME}},
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


def get_workflow_controlnet(positive, negative, reference_filename, cn_preset, seed=None):
    """Workflow с ControlNet Tile.
    Reference картинка → ImageScale → ControlNetApplyAdvanced → KSampler.
    LoRA остаётся, всё остальное идентично базовому workflow.
    v6.6: cn_preset — dict с keys strength/start/end (выбран в build_prompt)."""
    if seed is None:
        seed = int(time.time()) % 1000000000

    return {
        # ─── Базовые ноды (как в basic workflow) ───
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT_NAME}
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": 1, "height": 768, "width": 768}
        },
        "10": {
            "class_type": "LoraLoader",
            "inputs": {
                "clip": ["4", 1], "lora_name": "vocgems_jewelry_v2.safetensors",
                "model": ["4", 0], "strength_clip": 0.5, "strength_model": 0.5
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["10", 1], "text": positive}
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["10", 1], "text": negative}
        },

        # ─── ControlNet Tile цепочка ───
        # LoadImage → ImageScale (приводим к 768) → ControlNetApplyAdvanced
        # Tile НЕ требует препроцессора с обработкой контуров — он работает напрямую с RGB.
        "20": {
            "class_type": "LoadImage",
            "inputs": {"image": reference_filename}
        },
        "21": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["20", 0],
                "upscale_method": "lanczos",
                "width": 768,
                "height": 768,
                "crop": "center"
            }
        },
        "22": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": CONTROLNET_MODEL}
        },
        "23": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["6", 0],
                "negative": ["7", 0],
                "control_net": ["22", 0],
                "image": ["21", 0],
                "strength": cn_preset["strength"],
                "start_percent": cn_preset["start"],
                "end_percent": cn_preset["end"]
            }
        },

        # ─── KSampler берёт positive/negative из ControlNetApplyAdvanced ───
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 6,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["10", 0],
                "positive": ["23", 0],  # ← с ControlNet
                "negative": ["23", 1],  # ← с ControlNet
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "seed": seed,
                "steps": 30
            }
        },

        # ─── Output ───
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "vocgems", "images": ["8", 0]}
        }
    }


# ─── COMFYUI INTERACTION ──────────────────────────────────────────────────────

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


# ─── HANDLER ──────────────────────────────────────────────────────────────────

def handler(job):
    job_input = job.get("input", {})
    print(f"Job received: {job_input}", flush=True)

    if not wait_for_comfyui():
        return {"error": "ComfyUI failed to start"}

    positive, negative, cn_preset = build_prompt(job_input)

    # ─── Решаем: используем ControlNet или нет ───
    reference_url = job_input.get("reference_image_url", "").strip() if job_input.get("reference_image_url") else ""
    reference_filename = None
    use_controlnet = False

    cnet_model_path = f"/workspace/ComfyUI/models/controlnet/{CONTROLNET_MODEL}"
    cnet_available = os.path.exists(cnet_model_path)

    if reference_url and cnet_available:
        reference_filename = download_reference_image(reference_url)
        if reference_filename:
            use_controlnet = True
            print(f"=== Using ControlNet (strength={cn_preset['strength']}, end={cn_preset['end']}) ===", flush=True)
        else:
            print("=== Fallback to basic workflow (reference download failed) ===", flush=True)
    else:
        if not reference_url:
            print("=== No reference_image_url provided — using basic workflow ===", flush=True)
        if not cnet_available:
            print(f"=== ControlNet model missing at {cnet_model_path} — using basic workflow ===", flush=True)

    # ─── Строим workflow ───
    if use_controlnet:
        workflow = get_workflow_controlnet(positive, negative, reference_filename, cn_preset)
    else:
        workflow = get_workflow_basic(positive, negative)

    try:
        result = queue_prompt(workflow)
    except Exception as e:
        return {"error": f"Failed to queue prompt: {str(e)}"}

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        return {"error": "Failed to queue prompt"}

    print(f"Prompt queued: {prompt_id} (controlnet={use_controlnet})", flush=True)
    filename = wait_for_completion(prompt_id)

    if not filename:
        return {"error": "Generation timeout"}

    print(f"Generation complete: {filename}", flush=True)
    image_base64 = get_image(filename)

    return {
        "image": image_base64,
        "prompt_id": prompt_id,
        "filename": filename,
        "controlnet_used": use_controlnet
    }


print("Starting ComfyUI...", flush=True)
threading.Thread(target=start_comfyui, daemon=True).start()

runpod.serverless.start({"handler": handler})
