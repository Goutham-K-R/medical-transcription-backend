import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from utils import asr_bhashini, translate_bhashini, gemini_process
from slowapi import Limiter
from slowapi.util import get_remote_address

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class AudioInput(BaseModel):
    audioContent: str = Field(..., min_length=100, max_length=10_000_000, 
                            description="Base64 encoded audio content")
    language: str = Field(..., regex="^(en|hi|ml)$", 
                         description="Language code (en, hi, ml)")

# Health endpoints
@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "services": {
            "asr": True,
            "translation": True,
            "llm": True
        }
    }

# Main endpoint
@app.post("/transcribe")
@limiter.limit("5/minute")
async def transcribe(request: Request, input_data: AudioInput):
    try:
        # Input validation
        if not input_data.audioContent or len(input_data.audioContent) < 100:
            raise HTTPException(
                status_code=400, 
                detail="Audio content too short or empty"
            )
        
        logger.info(f"🌐 Language: {input_data.language}")
        logger.info(f"🔊 Audio length: {len(input_data.audioContent)} bytes")

        # Step 1: ASR
        original_transcript = asr_bhashini(
            input_data.audioContent, 
            input_data.language
        )
        logger.info(f"📝 ASR Output: {original_transcript}")

        # Step 2: Translation
        translated_text = original_transcript
        if input_data.language in ["ml", "hi"]:
            try:
                translated_text = translate_bhashini(
                    original_transcript, 
                    input_data.language
                )
                logger.info(f"🌍 Translated (en): {translated_text}")
            except Exception as e:
                logger.error(f"❌ Translation failed, using original: {e}")
                # Proceed with original text

        # Step 3: LLM processing
        try:
            result = gemini_process(original_transcript, translated_text)
            logger.info(f"✅ Gemini Result: {json.dumps(result, indent=2)}")
        except Exception as e:
            logger.error(f"❌ Gemini processing failed: {e}")
            result = {"extracted_terms": {}}

        # Response structure
        response_data = {
            "extracted_terms": result.get("extracted_terms", {}),
            "final_english_text": translated_text,
            "source_language": input_data.language,
            "raw_transcript": original_transcript,
            "success": True
        }

        return JSONResponse(content=response_data, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.critical(f"🔥 Internal Server Error: {e}", exc_info=True)
        return JSONResponse(
            content={
                "error": str(e),
                "success": False
            }, 
            status_code=500
        )