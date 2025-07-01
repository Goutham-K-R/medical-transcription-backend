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

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(input_data: AudioInput):
    try:
        logger.info(f"🌐 Language: {input_data.language}")

        # Step 1: Transcribe using Bhashini ASR
        original_transcript = asr_bhashini(input_data.audioContent, input_data.language)
        logger.info(f"📝 ASR Output: {original_transcript}")

        # Step 2: Translate if required
        if input_data.language in ["ml", "hi"]:
            try:
                translated_text = translate_bhashini(original_transcript, input_data.language)
                logger.info(f"🌍 Translated (en): {translated_text}")
            except Exception as e:
                logger.error(f"❌ Translation failed: {e}")
                translated_text = original_transcript
        else:
            translated_text = original_transcript

        # Step 3: LLM extraction
        result = gemini_process(original_transcript, translated_text)
        logger.info(f"✅ Gemini Result: {result}")

        # Updated response structure to match Flutter model
        response_data = {
            "data": {
                "extracted_terms": result.get("extracted_terms", {}),
                "final_english_text": translated_text or original_transcript or "No transcript found.",
            },
            "success": True
        }

        logger.info(f"✅ Final response to client: {response_data}")
        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        logger.critical(f"🔥 Internal Server Error: {e}", exc_info=True)
        return JSONResponse(content={
            "data": {
                "extracted_terms": {},
                "final_english_text": "",
            },
            "success": False,
            "error": str(e)
        }, status_code=500)