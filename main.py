from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from kokoro import KPipeline
import soundfile as sf
import torch
import uuid

app = FastAPI()

# 🇧🇷 Pipeline configurado para português
pipeline = KPipeline(lang_code='p')

# 🎯 Estrutura da requisição
class TTSRequest(BaseModel):
    text: str
    voice: str = 'af_heart'
    speed: float = 1.0


@app.post("/tts")
def generate_tts(request: TTSRequest):
    # 🔥 Gera áudio do texto completo
    generator = pipeline(
        request.text,
        voice=request.voice,
        speed=request.speed
    )

    # 🎧 Junta todos os segmentos de áudio gerados
    audio_segments = [audio for _, _, audio in generator]

    if not audio_segments:
        raise Exception("Nenhum áudio foi gerado. Verifique o texto de entrada.")

    combined_audio = torch.cat(audio_segments)

    # 💾 Salva como arquivo temporário WAV
    output_file = f"/tmp/{uuid.uuid4()}_output.wav"
    sf.write(output_file, combined_audio.numpy(), 24000)

    return FileResponse(output_file, media_type="audio/wav", filename="output.wav")


@app.get("/")
def read_root():
    return {"status": "Kokoro API is running and optimized!"}
