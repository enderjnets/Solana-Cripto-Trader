#!/usr/bin/env python3
"""Regenerate remaining 4 videos (longs + any missing) with Edge-TTS."""
import asyncio, json, subprocess
from pathlib import Path

WORKSPACE = Path('/home/enderj/.openclaw/workspace')
LOGO_PATH = WORKSPACE / 'videos/BIBLIOTECA/bittrader_logo.png'
EDGE_VOICE = 'es-MX-JorgeNeural'
EDGE_RATE = '+10%'

async def run():
    import edge_tts as etts
    from karaoke_subs import generate_karaoke_subs

    guiones = json.loads(Path('data/guiones_latest.json').read_text())
    scripts = guiones['scripts']

    for i, script in enumerate(scripts):
        title = script['title']
        vtype = script.get('type', 'short')
        out_file = Path(script.get('output_file', ''))
        if not out_file.exists():
            print(f'[{i+1}] Missing output: {out_file}')
            continue

        script_dir = out_file.parent
        text = script.get('script', '')
        v2_path = script_dir / out_file.name.replace('_final.mp4', '_v2_final.mp4')

        if v2_path.exists():
            print(f'[{i+1}] SKIP (already done): {v2_path.name}')
            continue

        print(f'[{i+1}] [{vtype.upper()}] {title[:40]}')

        # TTS
        mp3 = script_dir / out_file.name.replace('_final.mp4', '_edge.mp3')
        comm = etts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE)
        await comm.save(str(mp3))
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'csv=p=0', str(mp3)],
            capture_output=True, text=True
        )
        dur = float(r.stdout.strip()) if r.stdout.strip() else 60.0
        print(f'  Audio: {dur:.1f}s')

        # Subs
        generate_karaoke_subs(mp3, text, script_dir, vtype)

        # Intermediate (scaled)
        concat_path = script_dir / 'concat.txt'
        sub_path = script_dir / 'subtitles.ass'
        inter = script_dir / '_inter.mp4'

        if vtype == 'short':
            vf = (
                '[0:v]scale=1080:1920:force_original_aspect_ratio=increase,'
                'crop=1080:1920,avgblur=30[bg];'
                '[0:v]scale=1080:-2:force_original_aspect_ratio=decrease[fg];'
                '[bg][fg]overlay=(W-w)/2:(H-h)/2'
            )
            r1 = subprocess.run(
                ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                 '-i', str(concat_path),
                 '-filter_complex', vf,
                 '-t', str(dur + 1),
                 '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-an',
                 str(inter)],
                capture_output=True, timeout=60
            )
        else:
            r1 = subprocess.run(
                ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                 '-i', str(concat_path),
                 '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black',
                 '-t', str(dur + 1),
                 '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-an',
                 str(inter)],
                capture_output=True, timeout=60
            )

        if not inter.exists():
            print(f'  ❌ Intermediate failed')
            continue

        # Final assembly: video + audio + logo + subs
        logo_size = 240 if vtype == 'short' else 180
        sub_esc = str(sub_path).replace(':', r'\:')

        r2 = subprocess.run(
            ['ffmpeg', '-y',
             '-i', str(inter),
             '-i', str(mp3),
             '-i', str(LOGO_PATH),
             '-filter_complex',
             f'[2:v]scale={logo_size}:-1[logo];[0:v][logo]overlay=W-w-30:30:format=auto,ass=\'{sub_esc}\'',
             '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
             '-c:a', 'aac', '-b:a', '192k', '-ar', '44100',
             '-shortest', '-movflags', '+faststart',
             str(v2_path)],
            capture_output=True, text=True, timeout=120
        )

        inter.unlink(missing_ok=True)

        if v2_path.exists():
            print(f'  ✅ {v2_path.name} ({v2_path.stat().st_size/1024/1024:.1f}MB)')
        else:
            print(f'  ❌ Assembly failed: {r2.stderr[:200]}')

asyncio.run(run())
