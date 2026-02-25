import os
import cv2
import logging
import matplotlib.pyplot as plt
import re
from PIL import Image

from manga_translator.config import IMAGE_DIR, MODEL_PATH, FONT_PATH, TRANSLATED_DIR
from manga_translator.image_utils import reload_and_save_images, enhance_text_region
from manga_translator.detection import load_yolo_model, detect_text_regions, sort_bubbles
from manga_translator.ocr import mocr, validate_ocr_result, verify_japanese_text
from manga_translator.text_utils import clean_ocr_text, split_japanese_sentences, is_similar
from manga_translator.translation import clean_and_translate_text, post_process_translation, translator_local
from manga_translator.overlay import insert_translation, check_and_fix_truncated_text
from manga_translator.text_utils import manga_style_formatting
from difflib import SequenceMatcher


# Set up logging – more verbose for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("=== Starting MangaFlow processing ===")
    
    # Ensure directories exist
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(TRANSLATED_DIR, exist_ok=True)
    
    logging.info(f"Looking for images in: {os.path.abspath(IMAGE_DIR)}")
    all_files = os.listdir(IMAGE_DIR)
    logging.info(f"Found {len(all_files)} files/folders in images/: {all_files}")
    
    reload_and_save_images(IMAGE_DIR)
    logging.info("Image reload/save complete")

    # Load YOLO model
    logging.info(f"Loading YOLO model from: {MODEL_PATH}")
    try:
        model = load_yolo_model(MODEL_PATH)
        logging.info("YOLO model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        return

    # Filter valid images
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    logging.info(f"Found {len(image_files)} valid image files: {image_files}")
    
    if not image_files:
        logging.warning("No valid image files found in 'images/' folder. Add some .jpg, .png, or .webp manga pages and try again.")
        return

    for image_file in image_files:
        logging.info(f"--- Processing image: {image_file} ---")
        image_path = os.path.join(IMAGE_DIR, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            logging.warning(f"OpenCV could not read the image (possibly corrupted or unsupported format): {image_path}")
            continue
        logging.info(f"Image loaded successfully: {image.shape}")

        # Detect text regions
        logging.info("Running YOLO detection...")
        results = detect_text_regions(model, image_path)
        if results[0].boxes.cls.numel() == 0:
            logging.info("No text regions detected by YOLO")
        else:
            logging.info(f"Detected {len(results[0].boxes)} regions")

        # Process OCR on class 3 (text) only
        text_regions = []
        for box, cls_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            cls_id = int(cls_id)
            if cls_id != 3:
                continue
            x1, y1, x2, y2 = map(int, box)
            cropped = image[y1:y2, x1:x2]
            pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            text = mocr(pil_crop)
            if not (validate_ocr_result(text, cropped) and verify_japanese_text(text)):
                continue
            text_regions.append({'text': text, 'coords': (x1, y1, x2, y2)})
            logging.info(f"OCR successful on region {len(text_regions)}: '{text[:50]}...'")

        if not text_regions:
            logging.info("No valid Japanese text regions after OCR/validation – saving original as translated")
        
        # Clean and prepare sentences
        cleaned_text_regions = []
        for i, region in enumerate(text_regions):
            cleaned = clean_ocr_text(region['text'])
            formatted = split_japanese_sentences(cleaned)
            if formatted.strip():
                cleaned_text_regions.append({'id': i, 'text': formatted, 'coords': region['coords']})

        region_to_sentences = {}
        for region in cleaned_text_regions:
            sentences = [s.strip() for s in region['text'].split('\n') if s.strip()]
            if sentences:
                region_to_sentences[region['id']] = {'sentences': sentences, 'coords': region['coords']}

        # Deduplicate sentences
        all_sentences = []
        for region_id, data in region_to_sentences.items():
            for sentence in data['sentences']:
                all_sentences.append((region_id, sentence))

        unique_sentences = []
        for region_id, sentence in all_sentences:
            if not any(is_similar(sentence, existing[1]) for existing in unique_sentences):
                unique_sentences.append((region_id, sentence))

        logging.info(f"Unique sentences to translate: {len(unique_sentences)}")

        # === BATCH TRANSLATION FOR SPEED ===
        sentences_to_translate = [sentence for _, sentence in unique_sentences]
        batch_translations = []
        if sentences_to_translate:
            logging.info("Translating batch with NLLB...")
            results = translator_local.translate_text(sentences_to_translate)
            batch_translations = [res.text for res in results]
        
        translation_map = {}
        for idx, (region_id, sentence) in enumerate(unique_sentences):
            if idx >= len(batch_translations):
                continue
            raw_translation = batch_translations[idx]
            translation = manga_style_formatting(raw_translation)
            translation = re.sub(r'\s+([!?.,])', r'\1', translation)
            translation = re.sub(r'[\s\n]+', ' ', translation).strip()
            
            is_sfx = bool(re.search(r'[ドゴバキガ]{2,}', sentence))
            final = post_process_translation(translation, "sfx" if is_sfx else None)
            if final:
                translation_map[sentence] = final

        # Map back to regions
        region_translations = {}
        for region_id, region_data in region_to_sentences.items():
            sentences = region_data['sentences']
            translations = []
            for sentence in sentences:
                if not sentence.strip():
                    continue
                if sentence in translation_map:
                    translations.append(translation_map[sentence])
                else:
                    # Fallback similarity match
                    best = max(((orig, trans) for orig, trans in translation_map.items()), 
                               key=lambda x: SequenceMatcher(None, sentence, x[0]).ratio(), default=(None, None))
                    if best[0] and SequenceMatcher(None, sentence, best[0]).ratio() > 0.8:
                        translations.append(best[1])
            if translations:
                combined = ' '.join(translations)
                is_sfx_region = any(bool(re.search(r'[ドゴバキガ]{2,}', s)) for s in sentences)
                final_translation = post_process_translation(combined, "sfx" if is_sfx_region else None)
                region_translations[region_id] = {
                    'original': "\n".join(sentences),
                    'translation': final_translation,
                    'coords': region_data['coords']
                }

        logging.info(f"Final regions with translations: {len(region_translations)}")

        # Overlay and save
        translated_image = image.copy()
        for region_id, data in region_translations.items():
            if data['translation'].strip():
                translated_image = insert_translation(
                    translated_image,
                    data['coords'],
                    data['translation'],
                    font_path=FONT_PATH
                )

        translated_image = check_and_fix_truncated_text(translated_image, region_translations)

        output_path = os.path.join(TRANSLATED_DIR, f"translated_{image_file}")
        cv2.imwrite(output_path, translated_image)
        logging.info(f"Saved translated image: {output_path}")

        # Comparison plot
        comparison_path = os.path.join(TRANSLATED_DIR, f"comparison_{image_file}.png")
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB))
        plt.title("Translated")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=300)
        plt.close()
        logging.info(f"Saved comparison: {comparison_path}")

    logging.info("=== All processing complete ===")

if __name__ == "__main__":
    main()