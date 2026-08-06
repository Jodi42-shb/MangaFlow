#!/usr/bin/env python3

#For flat structure where images sit directly in a folder
"""
Docstring for main3_improved

Input
chin/
 └── [Fujisawa Tatsurou] Kaa-san Asa Made Daku yo!! ~Sachie Series Soushuuhen~ [English] {forbiddenfetish77} [Digital]/
      ├── 001.jpg
      ├── 002.jpg
      └── ...

Output
chin/
 └── [Fujisawa Tatsurou] Kaa-san Asa Made Daku yo!! ~Sachie Series Soushuuhen~ [English] {forbiddenfetish77} [Digital]/
      ├── 001.jpg
      ├── 002.jpg
      ├── ...
      └── translated/
           ├── translated_001.jpg
           └── ...
"""

import os
import shutil
import cv2
import logging
import re
from PIL import Image
from difflib import SequenceMatcher

from manga_translator.config import MODEL_PATH, FONT_PATH
from manga_translator.image_utils import reload_and_save_images
from manga_translator.detection import load_yolo_model, detect_text_regions
from manga_translator.ocr import mocr, validate_ocr_result, verify_japanese_text
from manga_translator.text_utils import (
    clean_ocr_text,
    split_japanese_sentences,
    is_similar,
    manga_style_formatting,
)
from manga_translator.translation import post_process_translation, translator_local
from manga_translator.overlay import insert_translation, check_and_fix_truncated_text

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -------------------------------------------------
# Process a folder of images directly
# -------------------------------------------------
def process_folder(folder_dir, model):
    folder_dir = folder_dir.strip('"').strip()

    if not os.path.isdir(folder_dir):
        logging.error(f"Invalid directory: {folder_dir}")
        return

    translated_dir = os.path.join(folder_dir, "translated")
    os.makedirs(translated_dir, exist_ok=True)

    logging.info(f"\n📘 Processing: {folder_dir}")
    logging.info(f"📁 Output to: {translated_dir}")

    reload_and_save_images(folder_dir)

    image_files = [
        f for f in os.listdir(folder_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    if not image_files:
        logging.warning("No images found in folder.")
        return

    for image_file in image_files:
        image_path = os.path.join(folder_dir, image_file)
        out_path = os.path.join(translated_dir, f"{image_file}")

        try:
            image = cv2.imread(image_path)

            if image is None:
                logging.warning(f"Failed to read image: {image_file} — copying original instead")
                shutil.copy2(image_path, out_path)
                continue

            # YOLO detection
            results = detect_text_regions(model, image_path)
            text_regions = []

            for box, cls_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                if int(cls_id) != 3:
                    continue

                x1, y1, x2, y2 = map(int, box)
                if x2 <= x1 or y2 <= y1:
                    # Degenerate/zero-area box from detection — skip just this box
                    continue
                cropped = image[y1:y2, x1:x2]
                pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                text = mocr(pil_crop)

                if not (
                    validate_ocr_result(text, cropped)
                    and verify_japanese_text(text)
                ):
                    continue

                text_regions.append({
                    "text": text,
                    "coords": (x1, y1, x2, y2)
                })

            if not text_regions:
                # No speech-bubble text detected on this page — still save it
                # untouched so page count matches the source folder.
                cv2.imwrite(out_path, image)
                logging.info(f"Saved (no text detected): {out_path}")
                continue

            # Clean & split
            cleaned_regions = []
            for i, region in enumerate(text_regions):
                cleaned = clean_ocr_text(region["text"])
                split = split_japanese_sentences(cleaned)
                if split.strip():
                    cleaned_regions.append({
                        "id": i,
                        "text": split,
                        "coords": region["coords"]
                    })

            region_map = {}
            for r in cleaned_regions:
                sentences = [s.strip() for s in r["text"].split("\n") if s.strip()]
                if sentences:
                    region_map[r["id"]] = {
                        "sentences": sentences,
                        "coords": r["coords"]
                    }

            # Deduplicate
            all_sentences = []
            for rid, data in region_map.items():
                for s in data["sentences"]:
                    all_sentences.append((rid, s))

            unique = []
            for rid, s in all_sentences:
                if not any(is_similar(s, u[1]) for u in unique):
                    unique.append((rid, s))

            translation_map = {}
            if unique:
                translations = translator_local.translate_text(
                    [s for _, s in unique]
                )

                for i, (_, sentence) in enumerate(unique):
                    raw = translations[i].text
                    styled = manga_style_formatting(raw)
                    styled = re.sub(r"\s+([!?.,])", r"\1", styled)
                    styled = re.sub(r"[\s\n]+", " ", styled).strip()

                    is_sfx = bool(re.search(r"[ドゴバキガ]{2,}", sentence))
                    final = post_process_translation(styled, "sfx" if is_sfx else None)

                    if final:
                        translation_map[sentence] = final

            region_translations = {}
            for rid, data in region_map.items():
                out = []
                for s in data["sentences"]:
                    if s in translation_map:
                        out.append(translation_map[s])
                    else:
                        best = max(
                            translation_map.items(),
                            key=lambda x: SequenceMatcher(None, s, x[0]).ratio(),
                            default=(None, None)
                        )
                        if best[0] and SequenceMatcher(None, s, best[0]).ratio() > 0.8:
                            out.append(best[1])

                if out:
                    region_translations[rid] = {
                        "translation": " ".join(out),
                        "coords": data["coords"]
                    }

            output_image = image.copy()
            for r in region_translations.values():
                output_image = insert_translation(
                    output_image,
                    r["coords"],
                    r["translation"],
                    font_path=FONT_PATH
                )

            output_image = check_and_fix_truncated_text(
                output_image,
                region_translations
            )

            cv2.imwrite(out_path, output_image)
            logging.info(f"Saved: {out_path}")

        except Exception:
            # Never let one bad page take down the whole batch. Log the full
            # traceback and fall back to saving the untouched original so the
            # page count still matches the source folder.
            logging.exception(f"Error processing {image_file} — saving original as fallback")
            try:
                if not os.path.exists(out_path):
                    shutil.copy2(image_path, out_path)
            except Exception:
                logging.exception(f"Fallback copy also failed for {image_file}")

# -------------------------------------------------
# Main interactive loop
# -------------------------------------------------
def main():
    logging.info("=== MangaFlow Batch Mode ===")
    logging.info(f"Loading YOLO model: {MODEL_PATH}")
    model = load_yolo_model(MODEL_PATH)

    while True:
        print("\n📂 Enter FOLDER path containing images directly (e.g. a manga volume folder)")
        print("Press Ctrl+Z (Windows) / Ctrl+D (Linux/macOS) to start\n")

        roots = []

        try:
            while True:
                line = input("> ").strip()
                if line:
                    roots.append(line)
        except EOFError:
            pass

        if not roots:
            logging.info("No input provided. Exiting.")
            break

        for root in roots:
            logging.info(f"\n🔍 Processing: {root}")
            process_folder(root, model)

        logging.info("\n✅ Batch complete. Enter more roots or Ctrl+Z again to exit.")

    logging.info("=== MangaFlow exited cleanly ===")


if __name__ == "__main__":
    main()
