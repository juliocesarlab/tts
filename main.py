from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from kokoro import KPipeline
import soundfile as sf
import torch
import uuid

app = FastAPI()

pipeline = KPipeline(lang_code='p')  # 🇧🇷 português brasileiro

class TTSRequest(BaseModel):
    text: str
    voice: str = 'af_heart'
    speed: float = 1.0

@app.post("/tts")
def generate_tts(request: TTSRequest):
    filename = f"/tmp/{uuid.uuid4()}.wav"
    generator = pipeline(
        request.text,
        voice=request.voice,
        speed=request.speed,
        split_pattern=r'\n+'
    )
    for i, (_, _, audio) in enumerate(generator):
        sf.write(filename, audio, 24000)
        break  # pega só o primeiro

    return FileResponse(filename, media_type="audio/wav", filename="output.wav")

@app.get("/")
def read_root():
    return {"status": "Kokoro API is running!"}
