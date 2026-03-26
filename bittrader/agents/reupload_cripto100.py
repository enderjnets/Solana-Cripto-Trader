#!/usr/bin/env python3
"""
Reupload script para edu_long_1774363903
- Sube video nuevo a YouTube
- Elimina video original (vckZR9iH7nM)
- Actualiza upload_queue.json
"""

import json
import requests
import os
import sys
import time

# ── Configuración ──────────────────────────────────────────
CREDS_PATH  = '/home/enderj/.openclaw/workspace/memory/youtube_credentials.json'
QUEUE_PATH  = '/home/enderj/.openclaw/workspace/bittrader/agents/data/upload_queue.json'
VIDEO_PATH  = '/home/enderj/.openclaw/workspace/bittrader/agents/output/2026-03-24/edu_long_1774363903/edu_long_1774363903_hybrid.mp4'
THUMB_PATH  = '/home/enderj/.openclaw/workspace/bittrader/agents/data/thumbnails/edu_long_1774363903_thumbnail.jpg'
SCRIPT_ID   = 'edu_long_1774363903'
ORIGINAL_ID = 'vckZR9iH7nM'

TITLE       = 'Cómo empezar en cripto con $100 — guía completa para principiantes 2026'
DESCRIPTION = ('Guía completa paso a paso para empezar en criptomonedas con solo $100. '
               'Sin tecnicismos, sin jerga — todo explicado desde cero para principiantes. '
               'Dónde comprar, qué comprar, cómo evitar los errores más comunes. '
               '#cripto #bitcoin #principiantes #tutorial #2026')
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
    'estrategia DCA cripto'
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
    # Persist
    creds['token'] = token
    with open(CREDS_PATH, 'w') as f:
        json.dump(creds, f, indent=2)
    print(f'[token] Refreshed OK: {token[:30]}...')
    return token


def upload_video(token):
    """Resumable upload via YouTube Data API v3"""
    file_size = os.path.getsize(VIDEO_PATH)
    print(f'[upload] Archivo: {VIDEO_PATH}')
    print(f'[upload] Tamaño: {file_size / 1024 / 1024:.1f} MB')

    # 1) Iniciar sesión resumable
    meta = {
        'snippet': {
            'title':       TITLE,
            'description': DESCRIPTION,
            'tags':        TAGS,
            'categoryId':  '27',   # Education
            'defaultLanguage': 'es'
        },
        'status': {
            'privacyStatus': 'public',
            'selfDeclaredMadeForKids': False
        }
    }

    init_url = 'https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status'
    init_r = requests.post(init_url,
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json; charset=UTF-8',
            'X-Upload-Content-Type': 'video/mp4',
            'X-Upload-Content-Length': str(file_size)
        },
        json=meta
    )
    if init_r.status_code not in (200, 201):
        raise RuntimeError(f'Upload init failed: {init_r.status_code} {init_r.text}')

    upload_url = init_r.headers['Location']
    print(f'[upload] Session URL obtenida')

    # 2) Subir en chunks de 10 MB
    chunk_size = 10 * 1024 * 1024
    uploaded   = 0
    new_id     = None

    with open(VIDEO_PATH, 'rb') as f:
        while uploaded < file_size:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            end = uploaded + len(chunk) - 1

            up_r = requests.put(upload_url,
                headers={
                    'Authorization':  f'Bearer {token}',
                    'Content-Type':   'video/mp4',
                    'Content-Range':  f'bytes {uploaded}-{end}/{file_size}'
                },
                data=chunk
            )

            if up_r.status_code in (200, 201):
                new_id = up_r.json().get('id')
                print(f'\n[upload] ✅ Completo! Nuevo ID: {new_id}')
                break
            elif up_r.status_code == 308:
                # Resume Incomplete — continúa
                rng = up_r.headers.get('Range', f'bytes=0-{end}')
                uploaded = int(rng.split('-')[1]) + 1
                pct = uploaded / file_size * 100
                print(f'[upload] {pct:.1f}% ({uploaded/1024/1024:.1f}/{file_size/1024/1024:.1f} MB)', end='\r')
            else:
                raise RuntimeError(f'Upload chunk error: {up_r.status_code} {up_r.text[:300]}')

    if not new_id:
        raise RuntimeError('No se obtuvo ID del video nuevo')

    return new_id


def set_thumbnail(token, video_id):
    file_size = os.path.getsize(THUMB_PATH)
    print(f'[thumb] Subiendo thumbnail ({file_size/1024:.0f} KB)...')

    # Initiate resumable thumbnail upload
    init_url = f'https://www.googleapis.com/upload/youtube/v3/thumbnails/set?videoId={video_id}&uploadType=resumable'
    init_r = requests.post(init_url,
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'image/jpeg',
            'X-Upload-Content-Type': 'image/jpeg',
            'X-Upload-Content-Length': str(file_size)
        }
    )
    if init_r.status_code not in (200, 201):
        print(f'[thumb] ⚠️  Init failed: {init_r.status_code} {init_r.text[:200]}')
        return False

    upload_url = init_r.headers.get('Location')
    if not upload_url:
        # Simple upload fallback
        simple_url = f'https://www.googleapis.com/upload/youtube/v3/thumbnails/set?videoId={video_id}&uploadType=media'
        with open(THUMB_PATH, 'rb') as f:
            data = f.read()
        r = requests.post(simple_url,
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'image/jpeg'
            },
            data=data
        )
        if r.status_code in (200, 201):
            print('[thumb] ✅ Thumbnail aplicado (simple)')
            return True
        else:
            print(f'[thumb] ⚠️  Simple upload failed: {r.status_code} {r.text[:200]}')
            return False

    with open(THUMB_PATH, 'rb') as f:
        data = f.read()
    r = requests.put(upload_url,
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'image/jpeg'
        },
        data=data
    )
    if r.status_code in (200, 201):
        print('[thumb] ✅ Thumbnail aplicado')
        return True
    else:
        print(f'[thumb] ⚠️  Upload failed: {r.status_code} {r.text[:200]}')
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
    with open(QUEUE_PATH, 'r') as f:
        queue = json.load(f)

    for item in queue:
        if item.get('script_id') == SCRIPT_ID:
            item['youtube_id']    = new_id
            item['video_id']      = new_id
            item['status']        = 'uploaded'
            item['needs_reupload'] = False
            item['error']         = None
            item['reupload_priority'] = None
            item['reuploaded_at'] = time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime())
            item['original_youtube_id'] = ORIGINAL_ID
            item['original_deleted']    = deleted_ok
            print(f'[queue] ✅ Actualizado: {SCRIPT_ID} → youtube_id={new_id}')
            break

    with open(QUEUE_PATH, 'w') as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)


def main():
    print('=' * 60)
    print('BitTrader Reupload — edu_long_1774363903')
    print('=' * 60)

    # 1. Verificar archivo
    if not os.path.exists(VIDEO_PATH):
        print(f'[ERROR] Archivo no encontrado: {VIDEO_PATH}')
        sys.exit(1)
    size_mb = os.path.getsize(VIDEO_PATH) / 1024 / 1024
    print(f'[check] ✅ Archivo OK: {size_mb:.1f} MB')

    # 2. Token
    creds = json.load(open(CREDS_PATH))
    token = refresh_token(creds)

    # 3. Subir nuevo video
    new_id = upload_video(token)

    # Esperar un momento para que YouTube procese
    print('[wait] Esperando 5s antes del thumbnail...')
    time.sleep(5)

    # 4. Thumbnail
    set_thumbnail(token, new_id)

    # 5. Eliminar original
    deleted_ok = delete_video(token, ORIGINAL_ID)

    # 6. Actualizar queue
    update_queue(new_id, deleted_ok)

    print()
    print('=' * 60)
    print(f'✅ COMPLETADO')
    print(f'   Nuevo YouTube ID : {new_id}')
    print(f'   Original {ORIGINAL_ID} eliminado: {deleted_ok}')
    print(f'   URL: https://youtu.be/{new_id}')
    print('=' * 60)


if __name__ == '__main__':
    main()
