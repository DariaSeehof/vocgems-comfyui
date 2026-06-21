"""
VOC Gems RunPod Serverless Handler
v6.7.5: v6.7.4 + жёсткая передача формы огранки (shape/cut).

Изменения от v6.7.4:
  - CUT_SHAPE_MAP: словарь форм огранки → (позитив с высоким весом,
    негатив чужих форм). Раньше форма шла сырым словом без веса
    ("octagon cut") и модель её игнорировала — octagon становился round.
  - resolve_cut(stone_cut): нормализует вход (Octagon / octagon cut /
    emerald → одна понятная модели фраза) и отдаёт ювелирные синонимы
    (octagon = emerald/step cut), которые SD1.5 понимает лучше "octagon".
  - В позитиве форма теперь с весом 1.4-1.5; в негативе блокируются
    конкурирующие формы (round/oval и т.п.), чтобы модель не усредняла.
  - Неизвестные формы — как раньше (умеренный вес, без негатива).

v6.7.4: v6.7.3 + переключатель LoRA (v2/v3) через job_input.

Изменения от v6.7.3:
  - Добавлена поддержка двух LoRA: v2 (прод, дефолт) и v3 (новая, тест).
  - При старте симлинкуются ОБЕ LoRA, если найдены на томе.
  - job_input["lora_version"] = "v3" → workflow берёт v3.
    По умолчанию (или "v2") → v2. Прод не меняется без явного флага.
  - LORA_FILES: словарь версия → имя файла.
  - resolve_lora_name(job_input) выбирает имя по флагу.
  - get_workflow_basic / get_workflow_controlnet принимают lora_name.

v6.7.3: STABLE BASELINE v6.7.2 + три точечные правки.

Изменения от v6.7.2:
  5. STONE_DEFAULT_COLORS: добавлены peridot, citrine, chrysoberyl,
     tsavorite, paraiba, zircon, kunzite, moonstone. Раньше эти камни
     попадали в is_unusual_color=True (defaults пустой → True), что
     давало неверное поведение build_color_negative.
  6. STONE_HARD_NEGATIVES: точечные негативы для камней, у которых
     SD1.5 имеет испорченный дефолт.
     - peridot: блокируем yellow/olive/brownish (Vivid Green → жёлтый)
     - citrine: блокируем brown/whisky/muddy yellow
  7. Anti-halo НЕ расширен на крупные камни.
     Тесты показали что это ломает Tourmaline Lagoon 12.75ct.

Изменения от v6.7.2 (унаследовано):
  3. BI-COLOR детектор — два цвета.
  4. Anti-halo для тёмных невыразительных камней (Grey/Black/Dark).

Изменения от v6.4 (унаследовано):
  1. METALS: одна фраза с весом 1.3.
  2. Дубль якоря изделия в конце позитива.

Что НЕ делаем (вынесено в roadmap):
  - Anti-warm-tone защита (v6.5) — давала ложные срабатывания
  - STONE_DESCRIPTORS / спец-якоря (v6.6) — давали "камень на постаменте"
  - WEAK_VISUAL_STONES пресеты Tile (v6.6) — для прозрачных камней
    отдельный pipeline в v6.8
  - Anti-floating-gem (v6.6.2) — фантомы — артефакт денойза
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

# ─── LoRA: две версии ───
# v2 — прод (дефолт). v3 — новая обученная LoRA, тестируется по флагу.
# Имя файла на томе → /runpod-volume/lora/<файл>
LORA_FILES = {
    "v2": "vocgems_jewelry_v2.safetensors",
    "v3": "vocgems_jewelry_v3.safetensors",
}
DEFAULT_LORA_VERSION = "v2"   # прод не меняется без явного флага lora_version


def resolve_lora_name(job_input):
    """Выбирает имя файла LoRA по флагу job_input['lora_version'].
    Неизвестное/пустое значение → дефолтная версия (v2, прод)."""
    version = str(job_input.get("lora_version", DEFAULT_LORA_VERSION)).strip().lower()
    if version not in LORA_FILES:
        version = DEFAULT_LORA_VERSION
    return LORA_FILES[version]

# ─── ControlNet параметры ───
# Tile передаёт ЦВЕТ и общую структуру референса, а не контуры.
# Это означает: цвет камня будет точно с фото, форма даст модели больше свободы,
# фон/рука/ткань референса будут размыты и не повлияют на финальный фон.
CONTROLNET_MODEL = "control_v11f1e_sd15_tile.pth"
CONTROLNET_STRENGTH = 0.3          # снижено: Tile должен подсказать ЦВЕТ, не весь контекст фото
CONTROLNET_START_PERCENT = 0.0
CONTROLNET_END_PERCENT = 0.35      # отпускаем рано: модель свободно делает белый фон и оправу
REFERENCE_FILENAME = "vocgems_reference.png"


def start_comfyui():
    global comfyui_process
    os.chdir("/workspace/ComfyUI")

    # ─── Симлинк LoRA (все версии: v2 прод + v3 тест) ───
    os.makedirs("/workspace/ComfyUI/models/loras", exist_ok=True)
    for ver, fname in LORA_FILES.items():
        lora_src = f"/runpod-volume/lora/{fname}"
        lora_dst = f"/workspace/ComfyUI/models/loras/{fname}"
        if os.path.exists(lora_src) and not os.path.exists(lora_dst):
            os.symlink(lora_src, lora_dst)
            print(f"LoRA linked ({ver}): {lora_src} -> {lora_dst}", flush=True)
        elif not os.path.exists(lora_src):
            print(f"WARNING: LoRA {ver} not found at {lora_src}", flush=True)

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
    # v6.7: одна фраза вес 1.3 (было: три фразы 1.5/1.4/1.3).
    # Металл больше не доминирует над камнем — цвет камня держится.
    "gold_750":   "(18k yellow gold band:1.3), warm gold setting",
    "white_gold": "(18k white gold band:1.3), polished silver-tone setting",
    "rose_gold":  "(18k rose gold band:1.3), warm pink gold setting",
    "platinum":   "(platinum band:1.3), polished cool-tone setting",
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
    # v6.7.3: добавлены камни из каталога VOC Gems, которых не было.
    # Без записи в словаре is_unusual_color возвращал True (defaults пустой)
    # для ЛЮБОГО цвета, что давало неверный вес 1.7 даже для дефолтных цветов.
    "peridot":     ["green"],
    "citrine":     ["yellow", "orange"],
    "chrysoberyl": ["yellow", "green"],
    "tsavorite":   ["green"],
    "paraiba":     ["blue", "green"],
    "zircon":      ["blue"],
    "kunzite":     ["pink"],
    "moonstone":   ["white", "colorless"],
}

UNUSUAL_COLOR_MARKERS = {
    "pastel", "light", "pale", "soft", "muted",
    "dark", "deep",
    "grey", "gray", "champagne", "peach", "salmon",
    "cognac", "honey", "lavender", "lilac", "mint",
    "teal", "olive", "neon", "smoky", "smokey",
}


# v6.7.3: Точечные негативы для конкретных типов камней,
# у которых SD1.5 имеет "испорченное" дефолтное представление.
# Применяются ВСЕГДА для данного типа камня, независимо от цвета.
STONE_HARD_NEGATIVES = {
    # Peridot: SD1.5 тянет в жёлто-зелёный/оливковый по умолчанию.
    # Для Vivid Green нам нужен чистый зелёный без жёлтых нот.
    "peridot": "(yellow peridot:1.5), (olive peridot:1.4), "
               "(yellowish gemstone:1.4), (brownish stone:1.4)",
    # Citrine: иногда сваливается в "оранжево-коричневый/виски" — для
    # чистого жёлтого/яркого блокируем коричневые тона.
    "citrine": "(brown citrine:1.3), (whisky color:1.3), (muddy yellow:1.3)",
}


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


# v6.7.2: Bi-Color детектор
# Срабатывает на маркеры в названии цвета или на тип камня "ametrine"
# (аметрин — природный bi-color аметист+цитрин).
BI_COLOR_MARKERS = {"bi-color", "bicolor", "bi color", "two-tone", "two tone",
                    "watermelon", "ametrine", "parti"}
BI_COLOR_STONE_TYPES = {"ametrine"}


def is_bi_color(stone_color, stone_type):
    color_lower = (stone_color or "").lower()
    type_lower = (stone_type or "").lower()
    if type_lower in BI_COLOR_STONE_TYPES:
        return True
    for marker in BI_COLOR_MARKERS:
        if marker in color_lower:
            return True
    return False


# v6.7.2: Anti-halo защита для серых/тёмных невыразительных камней.
# На таких камнях модель часто рисует halo pavé обрамление, которое
# выглядит как "второй камень внутри" или "призрак" — потому что
# яркие маленькие диаманты halo видны лучше чем тусклый центральный камень.
DARK_MUTED_COLOR_MARKERS = {"grey", "gray", "black", "smoky", "smokey",
                            "dark", "charcoal"}


def is_dark_muted_stone(stone_color, stone_type):
    color_lower = (stone_color or "").lower()
    for marker in DARK_MUTED_COLOR_MARKERS:
        if marker in color_lower:
            return True
    return False


# v6.7.5: Жёсткая передача ФОРМЫ ОГРАНКИ.
# Проблема: форма камня (octagon и т.п.) игнорировалась моделью — octagon
# превращался в round/oval. Причина: форма шла без веса и сырым словом,
# а SD1.5 лучше понимает ювелирные синонимы (octagon = emerald/step cut).
# Решение: нормализуем название огранки в понятную модели фразу +
# добавляем форму с высоким весом в позитив + блокируем чужие формы в негативе.
#
# Ключ — нормализованное входное значение (lower, без " cut").
# Значение: (phrase_for_positive, negatives_for_other_shapes)
CUT_SHAPE_MAP = {
    "octagon":      ("(emerald cut octagonal step-cut gemstone:1.5), (rectangular stone with cut corners:1.4), elongated rectangular faceted stone",
                     "round stone, oval stone, circular gem, pear shape, cushion shape, heart shape, marquise"),
    "emerald":      ("(emerald cut step-cut gemstone:1.5), (rectangular stone with cut corners:1.4), elongated rectangular faceted stone",
                     "round stone, oval stone, circular gem, pear shape, cushion shape, heart shape, marquise"),
    "emerald cut":  ("(emerald cut step-cut gemstone:1.5), (rectangular stone with cut corners:1.4), elongated rectangular faceted stone",
                     "round stone, oval stone, circular gem, pear shape, cushion shape, heart shape, marquise"),
    "baguette":     ("(baguette cut:1.5), (long narrow rectangular step-cut stone:1.4)",
                     "round stone, oval stone, circular gem, square stone, pear shape, cushion shape"),
    "round":        ("(round brilliant cut:1.5), (circular faceted gemstone:1.4)",
                     "rectangular stone, square stone, oval stone, pear shape, emerald cut, octagon, marquise"),
    "oval":         ("(oval cut:1.5), (elliptical faceted gemstone:1.4)",
                     "round circular stone, rectangular stone, square stone, pear shape, heart shape"),
    "cushion":      ("(cushion cut:1.5), (square stone with rounded corners:1.4), pillow-shaped faceted gemstone",
                     "round stone, rectangular stone, pear shape, marquise, emerald cut"),
    "pear":         ("(pear cut:1.5), (teardrop-shaped faceted gemstone:1.5), drop shape stone",
                     "round stone, oval stone, square stone, rectangular stone, heart shape, cushion"),
    "teardrop":     ("(pear cut:1.5), (teardrop-shaped faceted gemstone:1.5), drop shape stone",
                     "round stone, oval stone, square stone, rectangular stone, heart shape, cushion"),
    "marquise":     ("(marquise cut:1.5), (elongated pointed oval gemstone:1.4), navette shape",
                     "round stone, square stone, rectangular stone, cushion, emerald cut"),
    "heart":        ("(heart cut:1.5), (heart-shaped faceted gemstone:1.5)",
                     "round stone, oval stone, square stone, rectangular stone, pear shape"),
    "princess":     ("(princess cut:1.5), (square faceted gemstone with sharp corners:1.4)",
                     "round stone, oval stone, pear shape, emerald cut, cushion rounded corners"),
    "square":       ("(square cut gemstone:1.5), (square faceted stone:1.4)",
                     "round stone, oval stone, rectangular elongated stone, pear shape"),
    "radiant":      ("(radiant cut:1.5), (rectangular brilliant-cut gemstone with trimmed corners:1.4)",
                     "round stone, oval stone, pear shape, marquise, heart shape"),
    "asscher":      ("(asscher cut:1.5), (square step-cut gemstone with cropped corners:1.4)",
                     "round stone, oval stone, pear shape, marquise, elongated rectangle"),
    "trillion":     ("(trillion cut:1.5), (triangular faceted gemstone:1.5)",
                     "round stone, oval stone, square stone, rectangular stone, pear shape"),
    "round brilliant": ("(round brilliant cut:1.5), (circular faceted gemstone:1.4)",
                     "rectangular stone, square stone, oval stone, pear shape, emerald cut, octagon"),
}


def resolve_cut(stone_cut):
    """
    Возвращает (positive_phrase, shape_negative) по форме огранки.
    Принимает сырое значение (может быть 'Octagon', 'octagon cut', 'emerald' и т.п.).
    Если форма неизвестна — нейтральная фраза, без блокировки (как раньше).
    """
    raw = (stone_cut or "").strip().lower()
    if not raw:
        return ("(faceted cut:1.3)", "")
    # снимаем хвост " cut", чтобы 'octagon cut' и 'octagon' матчились одинаково
    key = raw[:-4].strip() if raw.endswith(" cut") else raw
    if key in CUT_SHAPE_MAP:
        return CUT_SHAPE_MAP[key]
    if raw in CUT_SHAPE_MAP:
        return CUT_SHAPE_MAP[raw]
    # неизвестная форма — отдаём как есть, с умеренным весом, без негатива
    return (f"({raw} cut:1.4), {raw} shape faceted gemstone", "")


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

    color_weight = 1.7 if is_unusual_color(stone_color, stone_type) else 1.5
    if stone_color:
        weighted_color = f"({stone_color}:{color_weight})"
        color_emphasis = f"{weighted_color} {stone_type}, {stone_color} colored gemstone, "
    else:
        color_emphasis = f"{stone_type}, "

    origin_part = f", from {stone_origin}" if stone_origin else ""
    # v6.7.5: форма огранки теперь с высоким весом + ювелирные синонимы.
    cut_phrase, cut_negative = resolve_cut(stone_cut)
    stone_desc = f"{color_emphasis}{stone_carat} carat, {cut_phrase}{origin_part}"

    metal_phrase = METALS.get(metal_key, METALS["gold_750"])
    style_phrase = STYLES.get(style_key, STYLES["modern"])
    diamonds_phrase = ", with small accent diamonds" if with_diamonds else ""
    wishes_phrase = f", {custom_wishes}" if custom_wishes else ""

    # v6.7.2: Bi-Color дескриптор. Подсказывает модели что камень
    # двухцветный с плавным переходом. Без этого модель усредняет в один цвет.
    bicolor_phrase = ""
    if is_bi_color(stone_color, stone_type):
        bicolor_phrase = (
            ", (bi-color gemstone with two distinct color zones:1.4), "
            "(visible color transition through the stone:1.3), "
            "split color gemstone"
        )

    positive = (
        f"vocgems jewelry, {weighted_anchor}, "
        f"(professional product catalog photography:1.5), "
        f"(commercial jewelry catalog:1.4), "
        f"{stone_desc}{bicolor_phrase}, {metal_phrase}, {style_phrase}{diamonds_phrase}{wishes_phrase}, "
        f"(pure white seamless background:1.6), "
        f"(plain white studio backdrop:1.5), "
        f"professional studio lighting, soft shadows, "
        f"8k resolution, sharp focus, "
        f"(isolated product shot:1.4), (no people:1.6), product only, "
        f"single piece centered, jewelry store catalog aesthetic, "
        # v6.7.1: дубль якоря изделия в конце удерживает композицию.
        # Без него Tile тащил "пустое кольцо без камня" с референсного фото
        # (где камень держится в пальцах отдельно от металла).
        f"({anchor}:1.5)"
    )

    type_neg = JEWELRY_NEG.get(jewelry_type, "")
    color_neg = build_color_negative(stone_color, stone_type)
    color_neg_part = f"{color_neg}, " if color_neg else ""
    metal_neg = METAL_NEG.get(metal_key, "")
    metal_neg_part = f"{metal_neg}, " if metal_neg else ""

    # v6.7.3: точечный негатив по типу камня (для peridot/citrine)
    stone_hard_neg = STONE_HARD_NEGATIVES.get(stone_type, "")
    stone_hard_neg_part = f"{stone_hard_neg}, " if stone_hard_neg else ""

    # v6.7.2: Anti-halo для тёмных невыразительных камней.
    # На сером/чёрном камне модель рисует halo pavé, которое создаёт
    # эффект "призрака второго камня" внутри/вокруг основного.
    anti_halo_part = ""
    if is_dark_muted_stone(stone_color, stone_type) and not with_diamonds:
        anti_halo_part = (
            "(halo setting:1.5), (pave halo:1.5), "
            "(accent diamonds around stone:1.4), "
            "(small stones inside main gem:1.5), "
            "(diamond cluster around center stone:1.4), "
        )

    # v6.7.5: блокируем чужие формы огранки, чтобы octagon не стал round/oval
    cut_neg_part = f"({cut_negative}:1.3), " if cut_negative else ""

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
        f"{stone_hard_neg_part}"
        f"{metal_neg_part}"
        f"{cut_neg_part}"
        f"{anti_halo_part}"
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

    print(f"=== PROMPT for {jewelry_type} ({stone_type}) ===", flush=True)
    print(f"POSITIVE: {positive}", flush=True)
    print(f"NEGATIVE: {negative}", flush=True)

    return positive, negative


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

def get_workflow_basic(positive, negative, lora_name, seed=None):
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
                "clip": ["4", 1], "lora_name": lora_name,
                "model": ["4", 0], "strength_clip": 0.5, "strength_model": 0.5
            }
        }
    }


def get_workflow_controlnet(positive, negative, reference_filename, lora_name, seed=None):
    """Workflow с ControlNet Canny.
    Reference картинка → Canny preprocessor → ControlNetApplyAdvanced → KSampler.
    LoRA остаётся, всё остальное идентично базовому workflow."""
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
                "clip": ["4", 1], "lora_name": lora_name,
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
                "strength": CONTROLNET_STRENGTH,
                "start_percent": CONTROLNET_START_PERCENT,
                "end_percent": CONTROLNET_END_PERCENT
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

    positive, negative = build_prompt(job_input)

    # ─── Выбор версии LoRA (v2 прод по умолчанию, v3 по флагу) ───
    lora_name = resolve_lora_name(job_input)
    print(f"=== Using LoRA: {lora_name} ===", flush=True)

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
            print(f"=== Using ControlNet (strength={CONTROLNET_STRENGTH}, end={CONTROLNET_END_PERCENT}) ===", flush=True)
        else:
            print("=== Fallback to basic workflow (reference download failed) ===", flush=True)
    else:
        if not reference_url:
            print("=== No reference_image_url provided — using basic workflow ===", flush=True)
        if not cnet_available:
            print(f"=== ControlNet model missing at {cnet_model_path} — using basic workflow ===", flush=True)

    # ─── Строим workflow ───
    if use_controlnet:
        workflow = get_workflow_controlnet(positive, negative, reference_filename, lora_name)
    else:
        workflow = get_workflow_basic(positive, negative, lora_name)

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
        "controlnet_used": use_controlnet,
        "lora_used": lora_name
    }


print("Starting ComfyUI...", flush=True)
threading.Thread(target=start_comfyui, daemon=True).start()

runpod.serverless.start({"handler": handler})
