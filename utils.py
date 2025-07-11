# utils.py

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# --- (The top part of the file with asr_bhashini and translate_bhashini remains unchanged) ---
ULCA_USER_ID = os.getenv("ULCA_USER_ID")
ASR_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/asr/v1/model/compute"
TRANSLATION_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/compute"

ASR_MODELS = { "ml": os.getenv("ULCA_ASR_MODEL_ML"), "hi": os.getenv("ULCA_ASR_MODEL_HI"), "en": os.getenv("ULCA_ASR_MODEL_EN")}
TRANSLATION_MODELS = { "hi": os.getenv("ULCA_TRANSLATION_MODEL_ID_HI"), "ml": os.getenv("ULCA_TRANSLATION_MODEL_ID_ML")}

def asr_bhashini(audio_base64, lang_code):
    model_id = ASR_MODELS.get(lang_code)
    if not model_id: raise ValueError(f"ASR Model ID not found for language: {lang_code}")
    payload = { "modelId": model_id, "task": "asr", "audioContent": audio_base64, "source": lang_code, "userId": ULCA_USER_ID }
    headers = {"Content-Type": "application/json"}
    response = requests.post(ASR_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    transcript = response.json().get("data", {}).get("source", "")
    if not transcript: raise ValueError("ASR service returned an empty transcript.")
    return transcript

def translate_bhashini(text, source_lang):
    model_id = TRANSLATION_MODELS.get(source_lang)
    if not model_id: raise ValueError(f"Translation Model ID not found for language: {source_lang}")
    payload = { "modelId": model_id, "task": "translation", "input": [{"source": text}], "config": {"language": {"sourceLanguage": source_lang, "targetLanguage": "en"}}, "userId": ULCA_USER_ID }
    headers = {"Content-Type": "application/json"}
    response = requests.post(TRANSLATION_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    output = response.json().get("output", [])
    if not output: raise ValueError("Translation service returned an empty output list.")
    translated_text = output[0].get("target", "")
    if not translated_text: raise ValueError("Translation service returned an empty result.")
    return translated_text
# --- END OF UNCHANGED PART ---


# --- THIS IS THE FINAL, DEFINITIVE "MASTER PROMPT" FUNCTION ---
def local_llm_process(original_text, translated_text):
    """
    Processes text with the final, most robust LLM prompt that is an expert
    at all clinical data extraction tasks, including complex dosage/frequency.
    """
    prompt = f"""
    You are an expert clinical data analyst with two key responsibilities:
    1.  **Correcting Errors**: You must identify and correct likely spelling or speech-to-text errors in the transcript (e.g., correct "Parasite Amol" to "Paracetamol").
    2.  **Extracting Data**: After mentally correcting the text, you must meticulously extract the information into a structured JSON object, ignoring any negated information.

    First, carefully read the definitions and strict rules for each category:
    - "Symptoms": Patient-reported issues (e.g., "headache", "shortness of breath").
    - "Symptom Triggers": The cause or timing of a symptom (e.g., "when climbing stairs").
    - "Medicine Names": Specific prescribed drug names.
        - **RULE 1 (CRITICAL)**: You MUST correct any misspellings or ASR errors (e.g., "Atorvasatin" becomes "Atorvastatin").
        - **RULE 2**: Do NOT include dosage, strength, or frequency here.
        - **RULE 3 (CRITICAL)**: Do NOT include generic forms like 'tablet', 'syrup', 'injection', or 'capsule'.
    - "Dosage & Frequency": The amount and timing of a dose (e.g., "500mg", "twice a day", "in the morning and evening", "one tablet after food"). # <-- ENHANCED DEFINITION
    - "Diseases / Conditions": Diagnosed or potential medical conditions (e.g., "Hypertension").
    - "Medical Procedures / Tests": Any ordered medical tests (e.g., "Blood test", "ECG").
    - "Duration": How long a treatment or symptom lasts (e.g., "for 3 days").
    - "Doctor's Instructions": Specific non-medication advice (e.g., "get plenty of rest", "drink warm water").

    ---
    **EXAMPLE OF YOUR TWO-STEP LOGIC IN ACTION:**

    *   **EXAMPLE INPUT TEXT (contains all tricky cases):**
        "Patient has a fever and it was mentioned he is taking a 500mg Parasite Amol tablet twice a day. He also takes Cof-Ex syrup in the morning and evening. Doctor asked about Atorvasatin, but patient said no. We need an ECG."

    *   **EXAMPLE JSON OUTPUT (shows correction, filtering, and all categories):** # <-- ENHANCED EXAMPLE
        {{
          "extracted_terms": {{
            "Symptoms": ["fever"],
            "Symptom Triggers": [],
            "Medicine Names": ["Paracetamol", "Cof-Ex"],
            "Dosage & Frequency": ["500mg", "twice a day", "in the morning and evening"],
            "Diseases / Conditions": [],
            "Medical Procedures / Tests": ["ECG"],
            "Duration": [],
            "Doctor's Instructions": []
          }}
        }}
    ---

    Now, apply this same two-step logic (Correct, then Extract) to the following text. Your response must be ONLY the valid JSON object.

    **Original Language Text:**
    "{original_text}"

    **English Translation:**
    "{translated_text}"

    Your JSON Output:
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:latest",
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=60
        )
        response.raise_for_status()
        generated_text = response.json()['response']
        
        if '```json' in generated_text:
            json_str = generated_text.split('```json\n')[1].split('```')[0]
            return json.loads(json_str)
        return json.loads(generated_text)

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Could not connect to local LLM: {e}")
    except json.JSONDecodeError:
        raise ValueError(f"LLM returned invalid JSON. Raw output: {generated_text}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred during LLM processing: {e}")