import os
import requests
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# --- Bhashini Configuration (Extended with all languages) ---
ULCA_USER_ID = os.getenv("ULCA_USER_ID")
ASR_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/asr/v1/model/compute"
TRANSLATION_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/compute"

# Extended ASR_MODELS with all supported languages
ASR_MODELS = {
    "ml": os.getenv("ULCA_ASR_MODEL_ML"),
    "hi": os.getenv("ULCA_ASR_MODEL_HI"),
    "en": os.getenv("ULCA_ASR_MODEL_EN"),
    "bn": os.getenv("ULCA_ASR_MODEL_BN"),
    "ta": os.getenv("ULCA_ASR_MODEL_TA"),
    "te": os.getenv("ULCA_ASR_MODEL_TE"),
    "kn": os.getenv("ULCA_ASR_MODEL_KN"),
    "gu": os.getenv("ULCA_ASR_MODEL_GU"),
    "pa": os.getenv("ULCA_ASR_MODEL_PA"),
    # "or": os.getenv("ULCA_ASR_MODEL_OR"),
    "as": os.getenv("ULCA_ASR_MODEL_AS"),
    "mr": os.getenv("ULCA_ASR_MODEL_MR"),
}

# Extended TRANSLATION_MODELS with all supported languages
TRANSLATION_MODELS = {
    "hi": os.getenv("ULCA_TRANSLATION_MODEL_ID_HI"),
    "ml": os.getenv("ULCA_TRANSLATION_MODEL_ID_ML"),
    "bn": os.getenv("ULCA_TRANSLATION_MODEL_ID_BN"),
    "ta": os.getenv("ULCA_TRANSLATION_MODEL_ID_TA"),
    "te": os.getenv("ULCA_TRANSLATION_MODEL_ID_TE"),
    "kn": os.getenv("ULCA_TRANSLATION_MODEL_ID_KN"),
    "gu": os.getenv("ULCA_TRANSLATION_MODEL_ID_GU"),
    "pa": os.getenv("ULCA_TRANSLATION_MODEL_ID_PA"),
    # "or": os.getenv("ULCA_TRANSLATION_MODEL_ID_OR"),
    "as": os.getenv("ULCA_TRANSLATION_MODEL_ID_AS"),
    "mr": os.getenv("ULCA_TRANSLATION_MODEL_ID_MR"),
}

# --- Gemini API Configuration (unchanged) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

def asr_bhashini(audio_base64, lang_code):
    # This function remains unchanged with your original logic
    model_id = ASR_MODELS.get(lang_code)
    if not model_id: 
        raise ValueError(f"ASR Model ID not found for language: {lang_code}")
    
    payload = {
        "modelId": model_id,
        "task": "asr",
        "audioContent": audio_base64,
        "source": lang_code,
        "userId": ULCA_USER_ID
    }
    
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(ASR_URL, headers=headers, json=payload)
    response.raise_for_status()
    
    transcript = response.json().get("data", {}).get("source", "")
    if not transcript:
        raise ValueError("ASR service returned an empty transcript.")
    
    return transcript

def translate_bhashini(text, source_lang):
    # This function remains unchanged with your original logic
    model_id = TRANSLATION_MODELS.get(source_lang)
    if not model_id:
        raise ValueError(f"Translation Model ID not found for language: {source_lang}")
    
    payload = {
        "modelId": model_id,
        "task": "translation",
        "input": [{"source": text}],
        "config": {
            "language": {
                "sourceLanguage": source_lang,
                "targetLanguage": "en"
            }
        },
        "userId": ULCA_USER_ID
    }
    
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(TRANSLATION_URL, headers=headers, json=payload)
    response.raise_for_status()
    
    output = response.json().get("output", [])
    if not output:
        raise ValueError("Translation service returned an empty output list.")
    
    translated_text = output[0].get("target", "")
    if not translated_text:
        raise ValueError("Translation service returned an empty result.")
    
    return translated_text

# --- GEMINI FUNCTION (Your original logic preserved) ---
def gemini_process(original_text, translated_text, translation_failed=False):
    """
    Processes text with the Gemini API, now with enhanced negation handling and
    resilience to translation failures.
    """
    base_prompt = """
You are an expert clinical data analyst with three key responsibilities:

1. **Correcting Errors**: You must identify and correct likely spelling or speech-to-text errors in the transcript (e.g., correct "Parasite Amol" to "Paracetamol").

2. **Extracting Data**: After mentally correcting the text, you must meticulously extract the information into a structured JSON object.

3. **Ignoring Negations**: You MUST ignore any symptoms, conditions, or medicines the patient explicitly denies having or taking.

First, carefully read the definitions and strict rules for each category:

- "Symptoms": Patient-reported issues (e.g., "headache", "shortness of breath").
- **RULE (CRITICAL)**: Do NOT include symptoms the patient says they DO NOT have.

- "Medicine Names": Specific prescribed drug names.
- **RULE 1 (CRITICAL)**: You MUST correct any misspellings or ASR errors (e.g., "Atorvasatin" becomes "Atorvastatin").
- **RULE 2**: Do NOT include dosage, strength, or frequency here.
- **RULE 3 (CRITICAL)**: If a brand name is present (e.g., "Cof-Ex"), extract only the brand name. Do NOT include the form (e.g., 'syrup', 'tablet').
- **RULE 4 (CRITICAL)**: If NO brand name is given, you MAY extract the general type of medicine (e.g., "cough syrup", "painkiller").
- **RULE 5 (CRITICAL)**: Do NOT include medicines the patient denies taking.

- "Dosage & Frequency": The amount and timing of a dose (e.g., "500mg", "twice a day").

- "Diseases / Conditions": Diagnosed or potential medical conditions (e.g., "Hypertension").
- **RULE (CRITICAL)**: Do NOT include conditions the patient denies having.

- "Medical Procedures / Tests": Any ordered medical tests (e.g., "Blood test", "ECG").

- "Duration": How long a treatment or symptom lasts (e.g., "for 3 days").

- "Doctor's Instructions": Specific non-medication advice (e.g., "get plenty of rest").

---

**EXAMPLES OF YOUR LOGIC IN ACTION:**

* **EXAMPLE 1 (Handling Negation):**
* INPUT TEXT: "Patient says no fever and no headache, but has a cough. He is taking Parasite Amol."
* YOUR JSON OUTPUT:
{
  "extracted_terms": {
    "Symptoms": ["cough"],
    "Medicine Names": ["Paracetamol"]
  }
}

* **EXAMPLE 2 (Brand Name Extraction):**
* INPUT TEXT: "He takes Cof-Ex syrup in the morning and evening for 3 days. We need an ECG."
* YOUR JSON OUTPUT:
{
  "extracted_terms": {
    "Medicine Names": ["Cof-Ex"],
    "Dosage & Frequency": ["in the morning and evening"],
    "Duration": ["for 3 days"],
    "Medical Procedures / Tests": ["ECG"]
  }
}

* **EXAMPLE 3 (Generic Medicine Extraction):**
* INPUT TEXT: "Doctor prescribed a cough syrup and some vitamin tablets."
* YOUR JSON OUTPUT:
{
  "extracted_terms": {
    "Medicine Names": ["cough syrup", "vitamin tablets"]
  }
}

---
"""

    # Conditionally add instructions based on whether translation worked
    if translation_failed:
        final_prompt = base_prompt + f"""
IMPORTANT: The English translation for the text below failed. You are receiving the text in its original language.
Do your best to understand the original text and extract the required entities.
CRITICAL RULE: Your final JSON output MUST contain only English terms.

**Original Language Text:**
"{original_text}"

Your JSON Output:
"""
    else:
        final_prompt = base_prompt + f"""
Now, apply your three-step logic (Correct, Extract, Ignore Negations) to the following text. Your response must be ONLY the valid JSON object.

**Original Language Text:**
"{original_text}"

**English Translation:**
"{translated_text}"

Your JSON Output:
"""

    try:
        model = genai.GenerativeModel(
            'gemini-1.5-flash-latest',
            generation_config={"response_mime_type": "application/json"}
        )
        
        response = model.generate_content(final_prompt)
        return json.loads(response.text)
        
    except json.JSONDecodeError:
        raise ValueError(f"Gemini API returned invalid JSON. Raw output: {response.text}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred with the Gemini API: {e}")
