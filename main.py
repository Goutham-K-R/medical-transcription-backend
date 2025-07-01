# main.py (Final Version - JSONResponse Fix Added)

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils import asr_bhashini, translate_bhashini, gemini_process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioInput(BaseModel):
    audioContent: str
    language: str

# ✅ Health Check
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Service is running!"}


# ✅ Transcription Route
@app.post("/transcribe")
async def transcribe(input_data: AudioInput):
    try:
        logger.info(f"Received request for language: {input_data.language}")

        original_transcript = asr_bhashini(input_data.audioContent, input_data.language)
        logger.info(f"Original transcript ({input_data.language}): {original_transcript}")

        if not original_transcript.strip():
            logger.warning("⚠️ Original transcript is empty.")
            return JSONResponse(content={
                "final_english_text": "",
                "extracted_terms": {}
            })

        english_transcript = original_transcript
        translation_failed = False

        if input_data.language in ['ml', 'hi']:
            try:
                english_transcript = translate_bhashini(original_transcript, input_data.language)
                logger.info(f"Translated transcript (en): {english_transcript}")
            except Exception as e:
                logger.error(f"Translation failed: {e}")
                translation_failed = True  # fallback to original language

        llm_result = gemini_process(original_transcript, english_transcript, translation_failed=translation_failed)
        logger.info("✅ Successfully processed with Gemini")

        response_data = {
            "final_english_text": original_transcript,
            "extracted_terms": llm_result.get("extracted_terms", {})
        }

        logger.info(f"✅ Sending response to UI: {response_data}")
        return JSONResponse(content=response_data)  # ✅ This guarantees a proper JSON response

    except ValueError as ve:
        logger.error(f"Value Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.critical(f"Unexpected Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
