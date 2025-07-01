# # main.py (Updated with health check and resilience flag)

# import logging
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from utils import asr_bhashini, translate_bhashini, gemini_process

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class AudioInput(BaseModel):
#     audioContent: str
#     language: str

# # --- ADDED: Health Check Endpoint for Render ---
# @app.get("/")
# def health_check():
#     """A simple endpoint to confirm the service is live."""
#     return {"status": "ok", "message": "Service is running!"}


# @app.post("/transcribe")
# async def transcribe(input_data: AudioInput):
#     try:
#         logger.info(f"Received request for language: {input_data.language}")

#         original_language_transcript = asr_bhashini(input_data.audioContent, input_data.language)
#         logger.info(f"Original transcript ({input_data.language}): {original_language_transcript}")

#         if not original_language_transcript.strip():
#             logger.warning("Original transcript is empty.")
#             return {"final_english_text": "", "extracted_terms": {}}

#         english_transcript_for_llm = original_language_transcript
#         translation_failed = False # --- ADDED: Initialize the flag ---

#         if input_data.language in ['ml', 'hi']:
#             try:
#                 english_transcript_for_llm = translate_bhashini(original_language_transcript, input_data.language)
#                 logger.info(f"Translated transcript for LLM (en): {english_transcript_for_llm}")
#             except Exception as e:
#                 logger.error(f"Translation failed: {e}. The LLM will proceed using the original text only.")
#                 english_transcript_for_llm = original_language_transcript
#                 translation_failed = True # --- ADDED: Set flag to True on failure ---

#         # --- MODIFIED: Pass the translation_failed flag to the Gemini function ---
#         llm_result = gemini_process(original_language_transcript, english_transcript_for_llm, translation_failed=translation_failed)
#         logger.info("Successfully processed with Gemini.")
        
#         final_response = {
#             "final_english_text": original_language_transcript,
#             "extracted_terms": llm_result.get("extracted_terms", {})
#         }

#         logger.info(f"Sending response to UI: {final_response}")
#         return final_response

#     except ValueError as ve:
#         logger.error(f"Value Error during transcription: {ve}")
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="An internal server error occurred.")

# main.py (Updated with explicit CORS preflight handler)

import logging
from fastapi import FastAPI, HTTPException, Response 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import asr_bhashini, translate_bhashini, gemini_process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Your existing CORS middleware is still important as a general rule.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class AudioInput(BaseModel):
    audioContent: str
    language: str

@app.get("/")
def health_check():
    """A simple endpoint to confirm the service is live."""
    return {"status": "ok", "message": "Service is running!"}


# --- THIS IS THE NEW BLOCK TO FIX THE CORS PREFLIGHT ERROR ---
@app.options("/transcribe")
async def transcribe_options():
    """
    An explicit OPTIONS endpoint to handle CORS preflight requests
    sent by the browser before the actual POST request to /transcribe.
    """
    logger.info("Received OPTIONS preflight request for /transcribe")
    # A 204 No Content response is standard and correct for a successful preflight.
    # The headers tell the browser what is allowed in the subsequent POST request.
    return Response(status_code=204, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS, DELETE, PUT",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    })
# --- END OF NEW BLOCK ---


@app.post("/transcribe")
async def transcribe(input_data: AudioInput):
    """
    The main endpoint for transcribing and processing audio.
    This code remains unchanged.
    """
    try:
        logger.info(f"Received request for language: {input_data.language}")

        original_language_transcript = asr_bhashini(input_data.audioContent, input_data.language)
        logger.info(f"Original transcript ({input_data.language}): {original_language_transcript}")

        if not original_language_transcript.strip():
            logger.warning("Original transcript is empty.")
            return {"final_english_text": "", "extracted_terms": {}}

        english_transcript_for_llm = original_language_transcript
        translation_failed = False

        if input_data.language in ['ml', 'hi']:
            try:
                english_transcript_for_llm = translate_bhashini(original_language_transcript, input_data.language)
                logger.info(f"Translated transcript for LLM (en): {english_transcript_for_llm}")
            except Exception as e:
                logger.error(f"Translation failed: {e}. The LLM will proceed using the original text only.")
                english_transcript_for_llm = original_language_transcript
                translation_failed = True

        llm_result = gemini_process(original_language_transcript, english_transcript_for_llm, translation_failed=translation_failed)
        logger.info("Successfully processed with Gemini.")
        
        final_response = {
            "final_english_text": original_language_transcript,
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