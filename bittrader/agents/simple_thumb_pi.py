#!/usr/bin/env python3
"""Simple thumbnail without AI - use solid gradient + text"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Create 1280x720 canvas
img = Image.new('RGB', (1280, 720), (30, 30, 40))  # Dark background
draw = ImageDraw.Draw(img)

# Add gradient (simple vertical)
for y in range(720):
    r = int(30 + (220 - 30) * y / 720)
    g = int(30 + (40 - 30) * y / 720)
    b = int(40 + (50 - 40) * y / 720)
    draw.line([(0, y), (1280, y)], fill=(r, g, b))

# Dark overlay at bottom for text
for y in range(450, 720):
    alpha = int(180 * (y - 450) / 270)
    color = (0, 0, 0, alpha)
    for x in range(1280):
        current = img.getpixel((x, y))
        img.putpixel((x, y), (
            int(current[0] * (1 - alpha/255) + 0 * alpha/255),
            int(current[1] * (1 - alpha/255) + 0 * alpha/255),
            int(current[2] * (1 - alpha/255) + 0 * alpha/255)
        ))

# Font
try:
    font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 75)
    font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    font_handle = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
except:
    font_big = ImageFont.load_default()
    font_sub = ImageFont.load_default()
    font_handle = ImageFont.load_default()

# Colors
RED = (220, 40, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Headline with black border
headline = "PI SE HUNDE\n-29%"
y = 480
line_height = 90

for line in headline.split('\n'):
    for ox in range(-12, 13):
        for oy in range(-12, 13):
            if abs(ox) + abs(oy) <= 16:
                draw.text((40 + ox, y + oy), line, font=font_big, fill=BLACK)
    draw.text((40, y), line, font=font_big, fill=RED)
    y += line_height

# Subtitle
subtitle = "En 24 horas"
for ox in range(-5, 6):
    for oy in range(-5, 6):
        if ox != 0 or oy != 0:
            draw.text((40, y + 10 + oy), subtitle, font=font_sub, fill=BLACK)
draw.text((40, y + 10), subtitle, font=font_sub, fill=WHITE)

# Handle
for ox in range(-3, 4):
    for oy in range(-3, 4):
        if ox != 0 or oy != 0:
            draw.text((980 + ox, 20 + oy), "@bittrader9259", font=font_handle, fill=BLACK)
draw.text((980, 20), "@bittrader9259", font=font_handle, fill=WHITE)

# Save
thumb_path = Path("/home/enderj/.openclaw/workspace/bittrader/agents/output/2026-03-14/short_1773470150_254/short_1773470150_254_thumbnail.jpg")
img.save(thumb_path, quality=95)

print(f"✅ Thumbnail guardado: {thumb_path}")
print(f"📏 1280x720")
