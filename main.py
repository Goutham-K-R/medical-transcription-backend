# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from utils import asr_bhashini, translate_bhashini, local_llm_process

# app = FastAPI()

# # Enable CORS for all origins
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class AudioInput(BaseModel):
#     audioContent: str  # Base64-encoded audio
#     language: str      # 'ml', 'hi', or 'en'

# @app.post("/transcribe")
# async def transcribe(input_data: AudioInput):
#     try:
#         transcript = asr_bhashini(input_data.audioContent, input_data.language)
        
#         if input_data.language in ['ml', 'hi']:
#             transcript = translate_bhashini(transcript, input_data.language)
        
#         result = local_llm_process(transcript)

#         return {
#             "status": "success",
#             "data": result
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": str(e)
#         }
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import asr_bhashini, translate_bhashini, local_llm_process

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioInput(BaseModel):
    audioContent: str  # Base64-encoded audio
    language: str      # 'ml', 'hi', or 'en'

@app.post("/transcribe")
async def transcribe(input_data: AudioInput):
    """
    Handles the end-to-end transcription pipeline:
    1. ASR (Speech-to-Text)
    2. Translation (if necessary)
    3. LLM processing for entity extraction
    """
    try:
        logger.info(f"Received request for language: {input_data.language}")

        # Step 1: Perform ASR
        transcript = asr_bhashini(input_data.audioContent, input_data.language)
        logger.info(f"Initial transcript ({input_data.language}): {transcript}")
        
        # Step 2: Translate to English if the source language is not English
        if input_data.language in ['ml', 'hi']:
            english_transcript = translate_bhashini(transcript, input_data.language)
            logger.info(f"Translated transcript (en): {english_transcript}")
        else:
            english_transcript = transcript
        
        # Step 3: Process the English transcript with the local LLM
        result = local_llm_process(english_transcript)
        logger.info("Successfully processed with LLM.")

        # On success, return the direct result from the LLM
        return result

    except ValueError as ve:
        # Handle known, client-fixable errors (e.g., bad language code, empty results)
        logger.error(f"Value Error during transcription: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Handle unexpected server-side or dependency errors
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")