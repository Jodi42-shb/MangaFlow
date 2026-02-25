import logging
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from manga_translator.text_utils import manga_style_formatting

# Load NLLB model once (local, offline)
MODEL_NAME = "facebook/nllb-200-distilled-600M"

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Loading NLLB model on {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# Language codes
SRC_LANG = "jpn_Jpan"
TGT_LANG = "eng_Latn"

# Use convert_tokens_to_ids for robust language forcing
forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

class LocalNLLBTranslator:
    def translate_text(self, texts, source_lang=None, target_lang=None, preserve_formatting=True):
        """
        Translate single string or list of strings using local NLLB.
        Mimics DeepL interface.
        """
        if not texts:
            return type('', (), {'text': ''})()

        if isinstance(texts, str):
            texts = [texts]

        tokenizer.src_lang = SRC_LANG

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=128,  # Better than max_length for generation
            num_beams=5,         # Slight quality boost
            early_stopping=True
        )

        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        class Result:
            def __init__(self, text):
                self.text = text

        return [Result(t) for t in translations]

# Global translator
translator_local = LocalNLLBTranslator()

def clean_and_translate_text(text, translator_local=translator_local, context=None):
    """
    Clean and translate single Japanese text to English using local NLLB.
    """
    if text.strip() in ['！', '。', '、', '．．．', '？']:
        return ""

    cleaned_text = text.strip()

    try:
        result = translator_local.translate_text(cleaned_text)
        translation = result[0].text

        translation = manga_style_formatting(translation)

        translation = re.sub(r'\s+([!?.,])', r'\1', translation)
        translation = re.sub(r'[\s\n]+', ' ', translation).strip()

        logging.info(f"Translated: {cleaned_text} -> {translation}")
        return translation

    except Exception as e:
        logging.error(f"Translation failed for {cleaned_text}: {e}")
        return ""

def post_process_translation(translation, text_type=None):
    """
    Same as before – unchanged.
    """
    if text_type is None:
        if bool(re.search(r'[ドゴバキガ]{2,}', translation)):
            text_type = "sfx"
        elif '!' in translation or '?' in translation:
            text_type = "emphasis"

    if text_type == "sfx":
        return f"*{translation.upper()}*"

    elif text_type == "emphasis":
        if '!' in translation and '?' in translation:
            return translation.upper() + "?!"
        elif '!' in translation:
            return translation.upper()
        else:
            return translation

    return translation