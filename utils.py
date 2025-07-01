import os
import requests
import json
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Bhashini Configuration ---
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

# --- Gemini API Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
genai.configure(api_key=GOOGLE_API_KEY)

# --- Helper Functions ---
def asr_bhashini(audio_base64, lang_code):
    """Convert speech to text using Bhashini ASR"""
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
    
    try:
        response = requests.post(
            ASR_URL, 
            headers=headers, 
            json=payload, 
            timeout=45
        )
        response.raise_for_status()
        
        transcript = response.json().get("data", {}).get("source", "")
        if not transcript:
            raise ValueError("ASR service returned an empty transcript.")
            
        return transcript
        
    except requests.exceptions.RequestException as e:
        logger.error(f"ASR request failed: {str(e)}")
        raise ValueError("ASR service unavailable")

def translate_bhashini(text, source_lang):
    """Translate text using Bhashini"""
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
    
    try:
        response = requests.post(
            TRANSLATION_URL, 
            headers=headers, 
            json=payload, 
            timeout=45
        )
        response.raise_for_status()
        
        output = response.json().get("output", [])
        if not output:
            raise ValueError("Translation service returned empty output")
            
        translated_text = output[0].get("target", "")
        if not translated_text:
            raise ValueError("Translation returned empty result")
            
        return translated_text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Translation request failed: {str(e)}")
        raise ValueError("Translation service unavailable")

def gemini_process(original_text, translated_text, translation_failed=False):
    """Process medical text with Gemini AI"""
    base_prompt = """
    [Previous prompt content remains exactly the same]
    """
    
    # Conditionally build final prompt
    if translation_failed:
        final_prompt = base_prompt + f"""
        IMPORTANT: The English translation failed. Process the original text:
        "{original_text}"
        Your JSON Output:
        """
    else:
        final_prompt = base_prompt + f"""
        Process this text:
        Original: "{original_text}"
        Translation: "{translated_text}"
        Your JSON Output:
        """
    
    try:
        model = genai.GenerativeModel(
            'gemini-1.5-flash-latest',
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.3
            }
        )
        
        response = model.generate_content(final_prompt)
        logger.debug(f"Gemini raw response: {response.text}")
        
        parsed = json.loads(response.text)
        
        # Validate response structure
        if not isinstance(parsed.get("extracted_terms", {}), dict):
            raise ValueError("Missing or invalid extracted_terms")
            
        # Ensure all categories exist
        required_categories = [
            "Symptoms", "Medicine Names", "Dosage & Frequency",
            "Diseases / Conditions", "Medical Procedures / Tests",
            "Duration", "Doctor's Instructions"
        ]
        
        for category in required_categories:
            if category not in parsed["extracted_terms"]:
                parsed["extracted_terms"][category] = []
                
        return parsed
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response: {e}")
        raise ValueError("Failed to process medical data")
    except Exception as e:
        logger.error(f"Gemini processing error: {str(e)}")
        raise