from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from kokoro import KPipeline
import soundfile as sf
import torch
import uuid
import re
import os
from pydub import AudioSegment

app = FastAPI()

# 🇧🇷 Pipeline em português brasileiro
pipeline = KPipeline(lang_code='p')

# ✅ Modelo da request
class TTSRequest(BaseModel):
    text: str
    voice: str = 'af_heart'
    speed: float = 1.0


# 🔥 Função para dividir texto em pedaços menores
def split_text(text, max_length=250):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


@app.post("/tts")
def generate_tts(request: TTSRequest):
    temp_files = []
    chunks = split_text(request.text)

    for idx, chunk in enumerate(chunks):
        generator = pipeline(
            chunk,
            voice=request.voice,
            speed=request.speed,
            split_pattern=r'\n+'
        )
        for i, (_, _, audio) in enumerate(generator):
            filename = f"/tmp/{uuid.uuid4()}_{idx}.wav"
            sf.write(filename, audio, 24000)
            temp_files.append(filename)
            break  # pega apenas o primeiro segmento por chunk

    # 🏗️ Combina todos os áudios em um único arquivo
    combined = AudioSegment.empty()
    for file in temp_files:
        sound = AudioSegment.from_wav(file)
        combined += sound

    output_file = f"/tmp/{uuid.uuid4()}_final.wav"
    combined.export(output_file, format="wav")

    # 🧹 Limpa arquivos temporários
    for file in temp_files:
        os.remove(file)

    return FileResponse(output_file, media_type="audio/wav", filename="output.wav")


@app.get("/")
def read_root():
    return {"status": "Kokoro API is running with split and merge!"}
