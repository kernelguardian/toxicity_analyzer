from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from google import genai
from google.genai import types
import uvicorn
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

origins = [
    "http://127.0.0.1:5501",
    "http://localhost:5501",
    "https://alanapi.kernelguardian.com",
]  # For dev only — lock this down in prod!

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic model to parse input JSON
class TextInput(BaseModel):
    text: str


# Initialize Gemini client once
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


@app.get("/")
async def root():
    return {"message": "Hello from FastAPI"}


@app.post("/check")
async def analyze_toxicity(input: TextInput):
    # Basic input validation (empty input)
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    model = "gemini-2.0-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input.text),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(
                text="""Analyse text content in double quotes and respond back only in json format by this guideline
REQUIREMENTS
1. toxicity score should be out of 10, follow rigid internal structure to get a score
2. highlighted text should be html supported highlighted problematic words with underline or bolded
    3. Anchor toxicity score based on this data [
    score 1: Well, that was brilliant, as usual.
    score 5: You're either completely clueless or just lazy.
    score 7: You’re such a pathetic excuse of a human it’s honestly disgusting.
    score 10: Go back to your rat-infested country, you useless dirty [slur].]

{
toxicityScore: 0,
categories: {
insult: 0.1,
threat: 0.5,
obscene: 0.3,
hate: 0.1,
sexual_explicit: 0.25,
identity_attack: 0.15,
profanity: 0.4,
toxicity: 0.6,
severe_toxicity: 0.05,
flirtation: 0.12,
},
highlightedText:''
}
"""
            ),
        ],
    )

    result = ""
    try:
        # Stream the response and accumulate text
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            result += chunk.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    try:
        parsed_result = json.loads(result)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse JSON: {str(e)}")

    # The result is expected to be JSON in string form, but the model might produce formatting errors.
    # You could attempt to parse JSON here to verify, or just return raw output for frontend to handle.
    return parsed_result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
