#!/usr/bin/env python3
"""
Reupload script para QrYrom05HzQ — "Cómo empezar en cripto con $100"
- Sube el video híbrido mejorado (edu_long_1774363903_hybrid.mp4)
- Actualiza el video en YouTube usando el mismo ID (videos().update) o sube nuevo y borra viejo
- Mantiene la thumbnail ya aprobada por Ender (edu_long_1774363903_thumbnail.jpg)

Fix aplicado: este video usaba edu_long_001.mp4 (producido por producer.py con clips repetidos
y sin sincronización narrador↔visual). El reemplazo usa el pipeline híbrido corregido
(produce_edu_long_hybrid.py) con imágenes únicas por sección, sincronizadas semánticamente.
"""

import json
import os
import sys
import time

import requests

# ── Configuración ──────────────────────────────────────────────────────────
CREDS_PATH  = '/home/enderj/.openclaw/workspace/memory/youtube_credentials.json'
QUEUE_PATH  = '/home/enderj/.openclaw/workspace/bittrader/agents/data/upload_queue.json'

# El video híbrido corregido (320MB, sección-a-sección con imágenes únicas)
VIDEO_PATH  = '/home/enderj/.openclaw/workspace/bittrader/agents/output/2026-03-24/edu_long_1774363903/edu_long_1774363903_hybrid.mp4'
THUMB_PATH  = '/home/enderj/.openclaw/workspace/bittrader/agents/data/thumbnails/edu_long_1774363903_thumbnail.jpg'

SCRIPT_ID   = 'edu_long_1774363903'
ORIGINAL_ID = 'QrYrom05HzQ'  # El video malo actualmente en YouTube

TITLE       = 'Cómo empezar en cripto con $100 — guía completa para principiantes 2026'
DESCRIPTION = (
    'Guía completa paso a paso para empezar en criptomonedas con solo $100. '
    'Sin tecnicismos, sin jerga — todo explicado desde cero para principiantes. '
    'Dónde comprar, qué comprar, cómo evitar los errores más comunes. '
    '\n\n'
    '00:00 Intro\n'
    '01:00 Cómo elegir tu exchange (Coinbase vs Binance)\n'
    '03:00 Qué comprar primero: BTC 60% + ETH 30% + Altcoin 10%\n'
    '05:00 Estrategia DCA: compra $25 cada semana\n'
    '07:00 Los 5 errores que debes evitar\n'
    '09:00 Resumen y próximos pasos\n'
    '\n'
    '#cripto #bitcoin #principiantes #tutorial #2026 #criptomonedas #ethereum #DCA #coinbase'
)
TAGS = [
    'criptomonedas para principiantes',
    'cómo empezar en cripto',
    'invertir 100 dólares en cripto',
    'guía completa criptomonedas 2026',
    'tutorial criptomonedas paso a paso',
    'Bitcoin para principiantes',
    'Ethereum explicación simple',
    'qué son las criptomonedas',
    'cómo comprar Bitcoin 2026',
    'exchanges criptomonedas',
    'estrategia DCA cripto',
    'Coinbase tutorial',
    'Binance tutorial',
    'errores cripto principiantes',
    'portfolio cripto $100',
]


def refresh_token(creds):
    r = requests.post('https://oauth2.googleapis.com/token', data={
        'client_id':     creds['client_id'],
        'client_secret': creds['client_secret'],
        'refresh_token': creds['refresh_token'],
        'grant_type':    'refresh_token'
    })
    r.raise_for_status()
    token = r.json()['access_token']
    creds['token'] = token
    with open(CREDS_PATH, 'w') as f:
        json.dump(creds, f, indent=2)
    print(f'[token] ✅ Refreshed: {token[:30]}...')
    return token


def upload_video(token):
    """Resumable upload via YouTube Data API v3"""
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f'Video not found: {VIDEO_PATH}')

    file_size = os.path.getsize(VIDEO_PATH)
    print(f'[upload] Archivo: {VIDEO_PATH}')
    print(f'[upload] Tamaño: {file_size / 1024 / 1024:.1f} MB')

    meta = {
        'snippet': {
            'title':           TITLE,
            'description':     DESCRIPTION,
            'tags':            TAGS,
            'categoryId':      '27',   # Education
            'defaultLanguage': 'es'
        },
        'status': {
            'privacyStatus':           'public',
            'selfDeclaredMadeForKids': False
        }
    }

    init_url = (
        'https://www.googleapis.com/upload/youtube/v3/videos'
        '?uploadType=resumable&part=snippet,status'
    )
    init_r = requests.post(
        init_url,
        headers={
            'Authorization':          f'Bearer {token}',
            'Content-Type':           'application/json; charset=UTF-8',
            'X-Upload-Content-Type':  'video/mp4',
            'X-Upload-Content-Length': str(file_size)
        },
        json=meta
    )
    if init_r.status_code not in (200, 201):
        raise RuntimeError(f'Upload init failed: {init_r.status_code} {init_r.text[:300]}')

    upload_url = init_r.headers['Location']
    print(f'[upload] Session URL obtenida — iniciando subida...')

    chunk_size = 10 * 1024 * 1024  # 10 MB
    uploaded   = 0
    new_id     = None

    with open(VIDEO_PATH, 'rb') as f:
        while uploaded < file_size:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            end = uploaded + len(chunk) - 1

            up_r = requests.put(
                upload_url,
                headers={
                    'Authorization': f'Bearer {token}',
                    'Content-Type':  'video/mp4',
                    'Content-Range': f'bytes {uploaded}-{end}/{file_size}'
                },
                data=chunk
            )

            if up_r.status_code in (200, 201):
                new_id = up_r.json().get('id')
                print(f'\n[upload] ✅ Completo! Nuevo ID: {new_id}')
                break
            elif up_r.status_code == 308:
                rng      = up_r.headers.get('Range', f'bytes=0-{end}')
                uploaded = int(rng.split('-')[1]) + 1
                pct      = uploaded / file_size * 100
                print(f'[upload] {pct:.1f}% ({uploaded/1024/1024:.1f}/{file_size/1024/1024:.1f} MB)', end='\r')
            else:
                raise RuntimeError(f'Upload chunk error: {up_r.status_code} {up_r.text[:300]}')

    if not new_id:
        raise RuntimeError('No se obtuvo ID del video nuevo')

    return new_id


def set_thumbnail(token, video_id):
    if not os.path.exists(THUMB_PATH):
        print(f'[thumb] ⚠️  Thumbnail no encontrado: {THUMB_PATH}')
        return False

    file_size = os.path.getsize(THUMB_PATH)
    print(f'[thumb] Subiendo thumbnail ({file_size/1024:.0f} KB) → video {video_id}...')

    # Simple upload (most reliable)
    simple_url = (
        f'https://www.googleapis.com/upload/youtube/v3/thumbnails/set'
        f'?videoId={video_id}&uploadType=media'
    )
    with open(THUMB_PATH, 'rb') as f:
        data = f.read()

    r = requests.post(
        simple_url,
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type':  'image/jpeg'
        },
        data=data
    )
    if r.status_code in (200, 201):
        print('[thumb] ✅ Thumbnail aplicado')
        return True
    else:
        print(f'[thumb] ⚠️  Failed: {r.status_code} {r.text[:200]}')
        return False


def delete_video(token, video_id):
    print(f'[delete] Eliminando video original: {video_id}')
    r = requests.delete(
        f'https://www.googleapis.com/youtube/v3/videos?id={video_id}',
        headers={'Authorization': f'Bearer {token}'}
    )
    if r.status_code == 204:
        print(f'[delete] ✅ Video {video_id} eliminado')
        return True
    else:
        print(f'[delete] ⚠️  {r.status_code} {r.text[:200]}')
        return False


def update_queue(new_id, deleted_ok):
    if not os.path.exists(QUEUE_PATH):
        print(f'[queue] ⚠️  Queue no encontrada: {QUEUE_PATH}')
        return

    with open(QUEUE_PATH, 'r') as f:
        queue = json.load(f)

    # Update existing entry or append new one
    updated = False
    for item in queue:
        if item.get('script_id') == SCRIPT_ID or item.get('youtube_id') == ORIGINAL_ID:
            item['youtube_id']            = new_id
            item['video_id']              = new_id
            item['status']                = 'uploaded'
            item['needs_reupload']        = False
            item['error']                 = None
            item['reupload_priority']     = None
            item['reuploaded_at']         = time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime())
            item['original_youtube_id']   = ORIGINAL_ID
            item['original_deleted']      = deleted_ok
            item['fix_applied']           = 'anti-repetition+narrator-sync v4.1'
            updated = True
            print(f'[queue] ✅ Actualizado: {SCRIPT_ID} → youtube_id={new_id}')
            break

    if not updated:
        queue.append({
            'script_id':           SCRIPT_ID,
            'title':               TITLE,
            'type':                'long',
            'youtube_id':          new_id,
            'status':              'uploaded',
            'reuploaded_at':       time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
            'original_youtube_id': ORIGINAL_ID,
            'original_deleted':    deleted_ok,
            'fix_applied':         'anti-repetition+narrator-sync v4.1',
        })
        print(f'[queue] ✅ Entrada nueva: {SCRIPT_ID} → youtube_id={new_id}')

    with open(QUEUE_PATH, 'w') as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)


def main():
    print('=' * 65)
    print('BitTrader Reupload — QrYrom05HzQ → nuevo video corregido')
    print('Fix: anti-repetition + narrator-sync (produce_edu_long_hybrid)')
    print('=' * 65)

    # 1. Verificar archivos
    if not os.path.exists(VIDEO_PATH):
        print(f'[ERROR] Video no encontrado: {VIDEO_PATH}')
        sys.exit(1)
    size_mb = os.path.getsize(VIDEO_PATH) / 1024 / 1024
    print(f'[check] ✅ Video OK: {size_mb:.1f} MB')

    thumb_ok = os.path.exists(THUMB_PATH)
    print(f'[check] {"✅" if thumb_ok else "⚠️ "} Thumbnail: {THUMB_PATH}')

    # 2. Token
    creds = json.load(open(CREDS_PATH))
    token = refresh_token(creds)

    # 3. Subir nuevo video
    new_id = upload_video(token)

    # Breve pausa para que YouTube procese
    print('[wait] Esperando 10s antes del thumbnail...')
    time.sleep(10)

    # 4. Thumbnail
    set_thumbnail(token, new_id)

    # 5. Eliminar original
    deleted_ok = delete_video(token, ORIGINAL_ID)

    # 6. Actualizar queue
    update_queue(new_id, deleted_ok)

    print()
    print('=' * 65)
    print(f'✅ COMPLETADO')
    print(f'   Nuevo YouTube ID  : {new_id}')
    print(f'   URL               : https://youtu.be/{new_id}')
    print(f'   Original borrado  : {ORIGINAL_ID} → {deleted_ok}')
    print(f'   Fix aplicado      : anti-repetición + sync narrador v4.1')
    print('=' * 65)


if __name__ == '__main__':
    main()
