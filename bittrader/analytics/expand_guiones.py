#!/usr/bin/env python3
"""
BitTrader Guion Expander
Usa MiniMax LLM para expandir guiones cortos en narraciones completas
"""
import json, requests, base64, os, subprocess, time
from pathlib import Path

WORKSPACE = Path("/home/enderj/.openclaw/workspace")
MINIMAX_KEY = json.load(open(WORKSPACE / "bittrader/keys/minimax.json"))["minimax_api_key"]
VOICE_ID = "presenter_male"
LOGO = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"

def expand_guion_with_llm(titulo, tipo="short"):
    """Usa MiniMax API como LLM para generar narración"""
    if tipo == "short":
        prompt = f"""Escribe una narración para un YouTube Short de 30-50 segundos sobre el tema: "{titulo}"

Reglas:
- Español neutro/latino (NO español de España)
- Empieza con un gancho fuerte (pregunta o afirmación impactante)
- Explica el concepto de forma simple
- Usa ejemplos concretos con números
- Termina con pregunta para comentarios
- NO uses emojis ni hashtags
- Solo el texto narrado, nada más
- Máximo 120 palabras"""
    else:
        prompt = f"""Escribe un guión de video largo (3-5 minutos) sobre: "{titulo}"

Reglas:
- Español neutro/latino
- Estructura: Hook → Problema → Explicación → Ejemplos → Conclusión → CTA
- Cada sección marcada con [SECCIÓN]
- Gancho fuerte en los primeros 10 segundos
- Ejemplos con números reales
- Termina con CTA para suscribirse
- NO uses emojis
- Máximo 600 palabras"""

    headers = {
        "Authorization": f"Bearer {MINIMAX_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "MiniMax-Text-01",
        "messages": [
            {"role": "system", "content": "Eres un experto en creación de contenido sobre criptomonedas y trading para YouTube. Tu estilo es directo, educativo y energético."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.8
    }
    
    resp = requests.post("https://api.minimax.io/v1/text/chatcompletion_v2",
                        headers=headers, json=payload, timeout=30)
    
    if resp.status_code == 200:
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return None

def generate_voice_minimax(text, output_path):
    """Genera audio con MiniMax TTS presenter_male"""
    headers = {"Authorization": f"Bearer {MINIMAX_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "speech-02-hd", "text": text, "stream": False,
        "voice_setting": {"voice_id": VOICE_ID, "speed": 1.0, "vol": 1.0, "pitch": 0},
        "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3"}
    }
    resp = requests.post("https://api.minimax.io/v1/t2a_v2", headers=headers, json=payload, timeout=60)
    data = resp.json()
    if data.get("base_resp", {}).get("status_code") == 0:
        audio = base64.b16decode(data["data"]["audio"].upper())
        Path(output_path).write_bytes(audio)
        return True
    return False

def generate_clip_minimax(prompt, output_path):
    """Genera clip de video con MiniMax"""
    headers = {"Authorization": f"Bearer {MINIMAX_KEY}", "Content-Type": "application/json"}
    
    # Crear task
    payload = {"prompt": prompt, "model": "MiniMax-Hailuo-2.3", "duration": 6, "resolution": "1080P"}
    resp = requests.post("https://api.minimax.io/v1/video_generation", headers=headers, json=payload, timeout=30)
    data = resp.json()
    task_id = data.get("task_id")
    if not task_id:
        return False
    
    # Poll hasta completar
    for _ in range(60):
        time.sleep(10)
        resp = requests.get("https://api.minimax.io/v1/query/video_generation",
                           headers=headers, params={"task_id": task_id})
        data = resp.json()
        if data.get("status") == "Success":
            file_id = data["file_id"]
            # Download
            resp2 = requests.get("https://api.minimax.io/v1/files/retrieve",
                               headers=headers, params={"file_id": file_id})
            url = resp2.json()["file"]["download_url"]
            video = requests.get(url)
            Path(output_path).write_bytes(video.content)
            return True
        elif data.get("status") == "Fail":
            return False
    return False

def get_duration(path):
    r = subprocess.run(["ffprobe","-v","quiet","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1",str(path)], capture_output=True, text=True)
    return float(r.stdout.strip())

def create_karaoke_subs(audio_path, ass_path):
    """Genera subtítulos karaoke sincronizados con Whisper"""
    import stable_whisper
    model = stable_whisper.load_model("small")
    result = model.transcribe(str(audio_path), language="es", word_timestamps=True)
    
    words = [{"word": w.word.strip(), "start": w.start, "end": w.end}
             for seg in result.segments for w in seg.words if w.word.strip()]
    
    def ts(s):
        h=int(s//3600); m=int((s%3600)//60); sec=s%60
        return f"{h}:{m:02d}:{sec:05.2f}"
    
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,66,&H0000FFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,2,0,1,3,1,2,60,60,130,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    groups = [words[i:i+5] for i in range(0, len(words), 5)]
    lines = [header]
    for g in groups:
        if not g: continue
        text = "".join(f"{{\\kf{int((w['end']-w['start'])*100)}}}{w['word']} " for w in g).strip()
        lines.append(f"Dialogue: 0,{ts(g[0]['start'])},{ts(g[-1]['end'])},Default,,0,0,0,,{text}")
    
    Path(ass_path).write_text("\n".join(lines), encoding="utf-8")

def assemble_short(clip_path, audio_path, ass_path, output_path):
    """Ensambla Short: loop clip + audio + subtitles + logo"""
    duration = min(get_duration(audio_path), 58)
    cmd = [
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(clip_path),
        "-i", str(audio_path), "-i", str(LOGO),
        "-filter_complex",
        f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[v];"
        f"[2:v]scale=100:100[logo];[v][logo]overlay=W-w-20:20[vl];"
        f"[vl]ass={ass_path}[vout]",
        "-map", "[vout]", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        "-t", str(duration), "-shortest", str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)

def produce_short(guion, output_dir):
    """Produce un Short completo desde un guion"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    name = guion["titulo"][:30].replace(" ", "_").replace("/", "")
    
    # 1. Expand guion if needed
    if guion.get("needs_expansion"):
        print(f"   📝 Expandiendo guión con LLM...")
        expanded = expand_guion_with_llm(guion["titulo"])
        if expanded:
            guion["guion"] = expanded
        else:
            print(f"   ⚠️ No se pudo expandir, usando título")
            guion["guion"] = guion["titulo"]
    
    # 2. Generate voice
    audio_path = output_dir / f"{name}_audio.mp3"
    print(f"   🎙️ Generando voz...")
    ok = generate_voice_minimax(guion["guion"], audio_path)
    if not ok:
        print(f"   ❌ Error generando voz")
        return None
    
    duration = get_duration(audio_path)
    if duration > 58:
        print(f"   ⚠️ Audio largo ({duration:.0f}s) — necesita guión más corto")
    
    # 3. Generate clip
    clip_path = output_dir / f"{name}_clip.mp4"
    clip_prompt = guion.get("clip_prompt", "Financial trading concept, modern digital aesthetic, cinematic lighting")
    print(f"   🎬 Generando clip MiniMax...")
    ok = generate_clip_minimax(clip_prompt, clip_path)
    if not ok:
        print(f"   ❌ Error generando clip")
        return None
    
    # 4. Generate karaoke subtitles
    ass_path = output_dir / f"{name}.ass"
    print(f"   📝 Generando subtítulos karaoke...")
    create_karaoke_subs(audio_path, ass_path)
    
    # 5. Assemble
    final_path = output_dir / f"{name}_final.mp4"
    print(f"   🎞️ Ensamblando...")
    assemble_short(clip_path, audio_path, ass_path, final_path)
    
    size = os.path.getsize(final_path) / 1024 / 1024
    print(f"   ✅ {final_path.name} ({size:.1f} MB, {min(duration,58):.0f}s)")
    
    return str(final_path)

def main():
    """Produce todos los shorts de los guiones generados"""
    # Find latest guiones
    guiones_dirs = sorted(Path("/home/enderj/.openclaw/workspace/bittrader").glob("semana_*"))
    if not guiones_dirs:
        print("❌ No hay guiones generados")
        return
    
    latest = guiones_dirs[-1]
    guiones_file = latest / "guiones.json"
    
    if not guiones_file.exists():
        print(f"❌ No existe {guiones_file}")
        return
    
    guiones = json.loads(guiones_file.read_text())
    shorts = [g for g in guiones if g["tipo"] == "short"]
    
    print(f"🎬 Produciendo {len(shorts)} Shorts\n")
    
    output_dir = latest / "shorts_final"
    results = []
    
    for i, guion in enumerate(shorts, 1):
        print(f"[{i}/{len(shorts)}] {guion['titulo'][:45]}...")
        result = produce_short(guion, output_dir)
        if result:
            results.append({"file": result, "titulo": guion["titulo"]})
        time.sleep(2)
    
    print(f"\n✅ {len(results)}/{len(shorts)} Shorts producidos!")
    
    # Save results
    (latest / "produced.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
