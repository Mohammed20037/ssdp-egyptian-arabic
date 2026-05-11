"""Stage 1: prompt generation.

Generates Egyptian Arabic text prompts via gpt-4o-mini. Each category
has its own Arabic system prompt with explicit dialect markers,
because generic "give me Egyptian Arabic" requests drift toward MSA.

Every candidate is cleaned, length-checked, Arabic-content-checked,
and hashed for dedup before being written to the manifest. The
manifest is append-only JSONL so a re-run continues from where the
last one stopped.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from .config import get_openai_key, load_config
from .utils import (
    append_jsonl,
    clean_prompt_text,
    looks_like_arabic,
    read_jsonl,
    setup_logging,
    text_hash,
    word_count,
)

log = setup_logging()
# Per-category guidance.
#
# Each entry tells the LLM:
#   - what register / situation
#   - what authentic Egyptian markers to use
#   - what to AVOID (MSA leakage is the main failure mode)
CATEGORY_INSTRUCTIONS: dict[str, str] = {
    "daily_conversation": (
        "محادثات يومية بالعامية المصرية - كلام صحاب، أهل، جيران، مواقف عادية. "
        "استخدم كلمات مصرية أصيلة زي: إزيك، إزاي، عايز، مش، دلوقتي، كده، "
        "ليه، إمتى، فين، أيوه، لأ، يلا، طب، يعني، خالص، أوي. "
        "تجنب الفصحى تماماً. كل جملة موقف مختلف."
    ),
    "customer_service": (
        "جمل من سياق خدمة عملاء أو مكالمات تليفونية بالعامية المصرية: "
        "استفسار عن طلب، شكوى، حجز، تأكيد، سؤال عن الفاتورة، إلخ. "
        "نبرة محايدة-مهذبة، مش رسمية فصحى. أمثلة: "
        "'أنا اتصلت قبل كده ومحدش رد عليا'، 'ممكن أعرف الطلب وصل فين؟'."
    ),
    "questions": (
        "أسئلة قصيرة بالعامية المصرية في مواضيع متنوعة (مواعيد، أماكن، "
        "أسعار، رأي، توضيح). استخدم أدوات الاستفهام المصرية (إزاي، إمتى، "
        "فين، ليه، إيه، مين، كام). جمل مش جمل خبرية."
    ),
    "code_switching": (
        "جمل عامية مصرية فيها كلمة أو اتنين إنجليزي مكتوبين بالحروف اللاتينية "
        "(زي ما المصريين بيتكلموا فعلاً). أمثلة: "
        "'ابعتلي الـ link على الواتساب'، "
        "'الـ meeting اتأجل لبكره'، "
        "'حجزت Uber وهييجي بعد دقيقتين'. "
        "الكلمة الإنجليزية لازم تكون في النص بحروف لاتينية، مش معربة."
    ),
    "numbers_dates": (
        "جمل عامية مصرية فيها أرقام، تواريخ، أسعار، أوقات، أو نسب. "
        "اكتب الأرقام بالعربي الهندي (٠١٢٣) أو الإنجليزي (0123) - نوّع. "
        "أمثلة: 'الميتنج الساعة 3 ونص'، 'دفعت 250 جنيه'، "
        "'المشروع هيبدأ ١٥ مارس'، 'خصم ٢٠٪ على المنتج ده'."
    ),
    "commands_requests": (
        "أوامر أو طلبات قصيرة بالعامية المصرية. صيغة الأمر المصري: "
        "'ابعت'، 'افتح'، 'هات'، 'سيب'، 'استنى'، 'يلا بينا'، 'متقلقش'. "
        "أو طلبات مهذبة: 'لو سمحت'، 'ممكن'، 'اعمل معروف'."
    ),
    "emotional": (
        "جمل عامية مصرية بنبرة عاطفية واضحة (فرح، غضب، إحباط، استغراب، "
        "حزن، حماس). استخدم تعبيرات مصرية أصيلة: 'يااااه'، 'لا مؤاخذة'، "
        "'والله العظيم'، 'يا سلام'، 'إيه ده'، 'أنا مش مصدق'، 'بجد؟'. "
        "نوّع المشاعر - مش كلها فرح ولا كلها غضب."
    ),
    "formal_news": (
        "جمل بنبرة رسمية أو إخبارية، لكن بأسلوب يقدر يقوله مذيع مصري - "
        "مش فصحى صرفة. سياقات: عناوين أخبار، إعلانات رسمية، تقارير. "
        "أمثلة: 'وزير التعليم أعلن النهارده عن...'، "
        "'الأرصاد توقعت طقس حار جداً الأسبوع الجاي'."
    ),
}


SYSTEM_PROMPT = (
    "أنت مساعد بيولّد جمل بالعامية المصرية الأصيلة، مش الفصحى، "
    "علشان تستخدم في تدريب نظام تعرف على الكلام. "
    "كل جملة لازم تكون طبيعية، زي ما المصريين بيتكلموا فعلاً، "
    "وقصيرة (من 3 لـ 25 كلمة). "
    "ترجع النتيجة JSON object واحد بالشكل ده بالظبط: "
    '{"prompts": ["جملة 1", "جملة 2", ...]} '
    "بدون أي نص إضافي. لا تحط أرقام أو ترقيم أو شرح."
)


# OpenAI call with retries
@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    reraise=True,
)
def _call_llm(
    client: OpenAI,
    model: str,
    temperature: float,
    category: str,
    n: int,
    min_words: int,
    max_words: int,
) -> list[str]:
    user_msg = (
        f"{CATEGORY_INSTRUCTIONS[category]}\n\n"
        f"ولّدلي {n} جملة مختلفة، كل جملة بين {min_words} و{max_words} كلمة. "
        "نوّع المواضيع والمواقف قدر الإمكان. لازم كل جملة فريدة ومش مكررة."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        log.warning("Bad JSON from LLM (%s); raw=%r", e, content[:200])
        return []
    items = data.get("prompts") or data.get("جمل") or []
    if not isinstance(items, list):
        return []
    return [str(x) for x in items if isinstance(x, (str, int, float))]


# Validation pipeline for a single candidate
def _validate(
    text: str,
    min_words: int,
    max_words: int,
    seen_hashes: set[str],
) -> tuple[bool, str]:
    """Returns (is_valid, reason_if_invalid)."""
    cleaned = clean_prompt_text(text)
    if not cleaned:
        return False, "empty_after_cleaning"
    wc = word_count(cleaned)
    if wc < min_words:
        return False, f"too_short_{wc}w"
    if wc > max_words:
        return False, f"too_long_{wc}w"
    if not looks_like_arabic(cleaned, min_ratio=0.3):
        return False, "not_arabic_enough"
    h = text_hash(cleaned)
    if h in seen_hashes:
        return False, "duplicate"
    return True, ""


# Plan: how many prompts per category, given totals already collected
def _plan_remaining(
    cfg: dict[str, Any],
    existing: list[dict[str, Any]],
) -> Counter:
    target_total = cfg["prompts"]["total_target"]
    proportions = cfg["prompts"]["categories"]
    targets = {cat: int(round(p * target_total)) for cat, p in proportions.items()}
    have = Counter(r["category"] for r in existing)
    remaining: Counter = Counter()
    for cat, target in targets.items():
        deficit = target - have.get(cat, 0)
        if deficit > 0:
            remaining[cat] = deficit
    return remaining


# Main entry point
def generate_prompts(config_path: str | None = None) -> Path:
    cfg = load_config(config_path)
    pcfg = cfg["prompts"]
    out_path = Path(cfg["paths"]["prompts_manifest"])

    existing = read_jsonl(out_path)
    seen_hashes: set[str] = {r["hash"] for r in existing}
    log.info("Loaded %d existing prompts from %s", len(existing), out_path)

    remaining = _plan_remaining(cfg, existing)
    if not remaining:
        log.info("Targets already met. Nothing to generate.")
        return out_path
    total_to_make = sum(remaining.values())
    log.info("Need %d more prompts across %d categories: %s",
             total_to_make, len(remaining), dict(remaining))

    client = OpenAI(api_key=get_openai_key())
    rng = random.Random(42)

    pbar = tqdm(total=total_to_make, desc="Generating prompts")
    invalid_reasons: Counter = Counter()
    consecutive_empty = 0

    while sum(remaining.values()) > 0:
        # Pick the category with the largest deficit; break ties randomly.
        cats_by_deficit = sorted(
            remaining.items(),
            key=lambda kv: (-kv[1], rng.random()),
        )
        category, deficit = cats_by_deficit[0]
        batch_n = min(pcfg["prompts_per_request"], deficit + 5)  # over-ask for filtering loss

        try:
            candidates = _call_llm(
                client=client,
                model=pcfg["model"],
                temperature=pcfg["temperature"],
                category=category,
                n=batch_n,
                min_words=pcfg["min_words"],
                max_words=pcfg["max_words"],
            )
        except Exception as e:
            log.error("LLM call failed for category=%s after retries: %s", category, e)
            consecutive_empty += 1
            if consecutive_empty >= 3:
                log.error("Aborting after 3 consecutive failures.")
                break
            continue

        added_this_round = 0
        for raw in candidates:
            ok, reason = _validate(
                raw,
                min_words=pcfg["min_words"],
                max_words=pcfg["max_words"],
                seen_hashes=seen_hashes,
            )
            if not ok:
                invalid_reasons[reason] += 1
                continue
            cleaned = clean_prompt_text(raw)
            h = text_hash(cleaned)
            record = {
                "id": h,
                "hash": h,
                "text": cleaned,
                "category": category,
                "word_count": word_count(cleaned),
                "source": "llm:" + pcfg["model"],
            }
            append_jsonl(out_path, record)
            seen_hashes.add(h)
            remaining[category] -= 1
            added_this_round += 1
            pbar.update(1)
            if remaining[category] <= 0:
                del remaining[category]
                break

        if added_this_round == 0:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                log.warning("3 consecutive batches yielded nothing valid; stopping early.")
                break
        else:
            consecutive_empty = 0

    pbar.close()
    final = read_jsonl(out_path)
    by_cat = Counter(r["category"] for r in final)
    log.info("Done. %d prompts total. By category: %s", len(final), dict(by_cat))
    if invalid_reasons:
        log.info("Filtered candidates: %s", dict(invalid_reasons))
    return out_path


if __name__ == "__main__":
    generate_prompts()
