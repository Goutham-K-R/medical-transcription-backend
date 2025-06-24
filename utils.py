# import os
# import requests
# from dotenv import load_dotenv

# load_dotenv()

# ULCA_USER_ID = os.getenv("ULCA_USER_ID")
# ASR_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/asr/v1/model/compute"
# TRANSLATION_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/compute"

# ASR_MODELS = {
#     "ml": os.getenv("ULCA_ASR_MODEL_ML"),
#     "hi": os.getenv("ULCA_ASR_MODEL_HI"),
#     "en": os.getenv("ULCA_ASR_MODEL_EN")
# }

# def asr_bhashini(audio_base64, lang_code):
#     model_id = ASR_MODELS.get(lang_code)
#     if not model_id:
#         raise ValueError("ASR Model ID not found for language: " + lang_code)

#     payload = {
#         "modelId": model_id,
#         "task": "asr",
#         "audioContent": audio_base64,
#         "source": lang_code,
#         "userId": ULCA_USER_ID
#     }

#     headers = {"Content-Type": "application/json"}

#     response = requests.post(ASR_URL, headers=headers, json=payload)
#     response.raise_for_status()
#     return response.json().get("data", {}).get("source", "")

# def translate_bhashini(text, source_lang):
#     if source_lang == 'hi':
#         model_id = os.getenv("ULCA_TRANSLATION_MODEL_ID_HI")
#     elif source_lang == 'ml':
#         model_id = os.getenv("ULCA_TRANSLATION_MODEL_ID_ML")
#     else:
#         raise ValueError("Unsupported language for translation: " + source_lang)

#     payload = {
#         "modelId": model_id,
#         "task": "translation",
#         "source": {"lang": source_lang, "text": text},
#         "userId": ULCA_USER_ID
#     }

#     headers = {"Content-Type": "application/json"}

#     response = requests.post(TRANSLATION_URL, headers=headers, json=payload)
#     response.raise_for_status()
#     return response.json().get("data", {}).get("target", "")
 


# import json

# def local_llm_process(text):
#     prompt = f"""
#     You are a medical assistant. Process the following clinical text and return structured JSON containing:
#     - final_english_text: The cleaned English version.
#     - extracted_terms: extracted medical info.

#     Example Output:
#     {{
#       "final_english_text": "<cleaned text>",
#       "extracted_terms": {{
#         "Medicine Names": [],
#         "Dosage & Frequency": [],
#         "Diseases / Conditions": [],
#         "Symptoms": [],
#         "Medical Procedures / Tests": [],
#         "Duration": [],
#         "Doctor's Instructions": []
#       }}
#     }}

#     Text to process:
#     \"\"\"{text}\"\"\"

#     Respond only in valid JSON format.
#     """

#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": "mistral:latest",
#             "prompt": prompt,
#             "stream": False
#         }
#     )

#     response.raise_for_status()
#     generated_text = response.json()['response']
    
#     try:
#         # Cleanly extract JSON from the model's response
#         start = generated_text.find('{')
#         end = generated_text.rfind('}') + 1
#         json_str = generated_text[start:end]
#         return json.loads(json_str)  # safe parsing
#     except Exception as e:
#         print(f"❌ Failed to parse JSON from LLM output: {e}")
#         return {"error": "Invalid LLM JSON", "raw_output": generated_text}
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# --- Bhashini API Configuration ---
ULCA_USER_ID = os.getenv("ULCA_USER_ID")
ASR_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/asr/v1/model/compute"
TRANSLATION_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/compute"

ASR_MODELS = {
    "ml": os.getenv("ULCA_ASR_MODEL_ML"),
    "hi": os.getenv("ULCA_ASR_MODEL_HI"),
    "en": os.getenv("ULCA_ASR_MODEL_EN")
}

TRANSLATION_MODELS = {
    "hi": os.getenv("ULCA_TRANSLATION_MODEL_ID_HI"),
    "ml": os.getenv("ULCA_TRANSLATION_MODEL_ID_ML")
}

def asr_bhashini(audio_base64, lang_code):
    """Performs Automatic Speech Recognition using Bhashini API."""
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

    response = requests.post(ASR_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    
    transcript = response.json().get("data", {}).get("source", "")
    if not transcript:
        raise ValueError("ASR service returned an empty transcript.")
    return transcript

# --- THIS IS THE CORRECTED FUNCTION ---
def translate_bhashini(text, source_lang):
    """Translates text to English using Bhashini API."""
    model_id = TRANSLATION_MODELS.get(source_lang)
    if not model_id:
        raise ValueError(f"Translation Model ID not found for language: {source_lang}")

    # The payload for the v0/model/compute endpoint has a different structure.
    payload = {
        "modelId": model_id,
        "task": "translation",
        "input": [
            {
                "source": text
            }
        ],
        "config": {
            "language": {
                "sourceLanguage": source_lang,
                "targetLanguage": "en"
            }
        },
        "userId": ULCA_USER_ID
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(TRANSLATION_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    # The response structure is also different. The result is inside the 'output' key.
    output = response.json().get("output", [])
    if not output:
        raise ValueError("Translation service returned an empty output list.")

    translated_text = output[0].get("target", "")
    if not translated_text:
        raise ValueError("Translation service returned an empty result.")
        
    return translated_text

def local_llm_process(text):
    """Processes text with a local LLM to extract structured medical data."""

    prompt = f"""
    You are an expert clinical data analyst with two key responsibilities:
    1.  **Correcting Errors**: You must identify and correct likely spelling or speech-to-text errors in the transcript (e.g., correct "Parasite Amol" to "Paracetamol").
    2.  **Extracting Data**: You must meticulously extract the corrected information into a structured JSON object.

    First, carefully read the definitions and rules for each category:
    - "final_english_text": A cleaned, coherent summary of the conversation, containing the *corrected* information.
    - "Symptoms": Patient-reported issues (e.g., "headache", "shortness of breath when walking").
    - "Medicine Names": Specific prescribed drug names.
        - **RULE 1**: Correct any misspellings or ASR errors (e.g., "Atorvasatin" becomes "Atorvastatin").
        - **RULE 2**: Do NOT include dosage, strength, or frequency.
        - **RULE 3 (CRITICAL)**: Do NOT include generic forms like 'tablet', 'syrup', 'injection', 'capsule', or 'ointment'.
    - "Dosage & Frequency": The amount and timing of a dose (e.g., "500mg", "twice a day").
    - "Diseases / Conditions": Diagnosed or potential medical conditions (e.g., "Hypertension").
    - "Medical Procedures / Tests": Any ordered medical tests (e.g., "Blood test", "ECG").
    - "Duration": How long a treatment or symptom lasts (e.g., "for 3 days").
    - "Doctor's Instructions": Specific non-medication advice (e.g., "get plenty of rest").

    ---
    Here is an example of how to perform the task correctly:

    EXAMPLE INPUT TEXT (contains errors):
    "Patient has a fever and it was mentioned he is taking Dolo and also Parasite Amol. And a bottle of Cof-Ex syrup. We need an ECG."

    EXAMPLE JSON OUTPUT (shows correction and filtering):
    {{
      "final_english_text": "The patient has a fever and is taking Dolo and Paracetamol. He is also taking Cof-Ex syrup. An ECG is required.",
      "extracted_terms": {{
        "Symptoms": ["fever"],
        "Medicine Names": ["Dolo", "Paracetamol", "Cof-Ex"],
        "Dosage & Frequency": [],
        "Diseases / Conditions": [],
        "Medical Procedures / Tests": ["ECG"],
        "Duration": [],
        "Doctor's Instructions": []
      }}
    }}
    ---

    Now, apply this same two-step logic (Correct, then Extract) to the following text. Your response must be ONLY the valid JSON object.

    Text to process:
    \"\"\"{text}\"\"\"
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
        
        # Clean potential markdown formatting from the LLM's response
        if '```json' in generated_text:
            json_str = generated_text.split('```json\n')[1].split('```')[0]
            return json.loads(json_str)
        return json.loads(generated_text)

    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to local LLM: {e}")
        raise ValueError("The local language model is not reachable.")
    except json.JSONDecodeError:
        print(f"❌ Failed to parse JSON from LLM output. Raw output:\n{generated_text}")
        raise ValueError("The language model returned an invalid JSON format.")
    except Exception as e:
        print(f"❌ An unexpected error occurred during LLM processing: {e}")
        raise