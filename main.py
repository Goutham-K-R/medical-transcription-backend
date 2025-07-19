import logging
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Literal
from utils import asr_bhashini, translate_bhashini, gemini_process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioInput(BaseModel):
    audioContent: str
    language: Literal["ml", "hi", "en", "bn", "ta", "te", "kn", "gu", "pa", "as", "mr"]
    
    @validator('language')
    def validate_language(cls, v):
        supported_languages = ["ml", "hi", "en","bn", "ta", "te", "kn", "gu", "pa", "as", "mr"]
        if v not in supported_languages:
            raise ValueError(f"Language '{v}' is not supported. Supported languages: {supported_languages}")
        return v

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Service is running!"}

@app.get("/supported-languages")
def get_supported_languages():
    return {
        "supported_languages": {
            "ml": "Malayalam",
            "hi": "Hindi", 
            "en": "English",
            "bn": "Bengali",
            "ta": "Tamil",
            "te": "Telugu",
            "kn": "Kannada",
            "gu": "Gujarati",
            "pa": "Punjabi",
            # "or": "Odia",
            "as": "Assamese",
            "mr": "Marathi"
        }
    }

@app.options("/transcribe")
async def transcribe_options():
    logger.info("Received OPTIONS preflight request for /transcribe")
    return Response(status_code=204, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS, DELETE, PUT",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    })

@app.post("/transcribe")
async def transcribe(input_data: AudioInput):
    try:
        logger.info(f"Received request for language: {input_data.language}")
        
        original_language_transcript = asr_bhashini(input_data.audioContent, input_data.language)
        logger.info(f"Original transcript ({input_data.language}): {original_language_transcript}")
        
        if not original_language_transcript.strip():
            logger.warning("Original transcript is empty.")
            return {"final_english_text": "", "extracted_terms": {}}
        
        english_transcript_for_llm = original_language_transcript
        translation_failed = False
        
        # Updated to support all languages except English
        if input_data.language in ['ml', 'hi', 'bn', 'ta', 'te', 'kn', 'gu', 'pa', 'or', 'as', 'mr']:
            try:
                print('wefjiklermkfmklew')
                print(original_language_transcript)
                english_transcript_for_llm = translate_bhashini(original_language_transcript, input_data.language)
                logger.info(f"Translated transcript for LLM (en): {english_transcript_for_llm}")
            except Exception as e:
                logger.error(f"Translation failed: {e}. The LLM will proceed using the original text only.")
                english_transcript_for_llm = original_language_transcript
                translation_failed = True
        
        llm_result = gemini_process(original_language_transcript, english_transcript_for_llm, translation_failed=translation_failed)
        logger.info("Successfully processed with Gemini.")
        
        final_response = {
            "final_english_text": english_transcript_for_llm,
            "extracted_terms": llm_result.get("extracted_terms", {})
        }
        
        logger.info(f"Sending response to UI: {final_response}")
        return final_response
        
    except ValueError as ve:
        logger.error(f"Value Error during transcription: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000, host="localhost")
