from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.colors import HexColor
from pdfrw import PdfReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
import io, os

CARD_W_MM = 88.9
CARD_H_MM = 50.8
PAGE_W, PAGE_H = A4
SRC_W = 595.92
SRC_H = 842.88
scale_x = PAGE_W / SRC_W
scale_y = PAGE_H / SRC_H
PADDING_MM = 8
GAP_MM = 5
CARDS_DIR = '/home/enderj/.openclaw/workspace/cards'

def card_positions():
    positions = []
    for row in range(5):
        for col in range(2):
            x_mm = PADDING_MM + col * (CARD_W_MM + GAP_MM)
            y_top_mm = PADDING_MM + row * (CARD_H_MM + GAP_MM)
            y_bottom_mm = 297 - y_top_mm - CARD_H_MM
            x_pts = x_mm * mm * scale_x
            y_pts = y_bottom_mm * mm * scale_y
            positions.append((x_pts, y_pts))
    return positions

positions = card_positions()
CARD_W_PTS = CARD_W_MM * mm * scale_x
CARD_H_PTS = CARD_H_MM * mm * scale_y

def draw_cut_marks(c, x, y, w, h, size=3*mm, offset=2*mm):
    c.saveState()
    c.setStrokeColor(HexColor('#555555'))
    c.setLineWidth(0.4)
    # Top-left corner
    c.line(x-offset-size, y+h, x-offset, y+h)
    c.line(x, y+h+offset, x, y+h+offset+size)
    # Top-right corner
    c.line(x+w+offset, y+h, x+w+offset+size, y+h)
    c.line(x+w, y+h+offset, x+w, y+h+offset+size)
    # Bottom-left corner
    c.line(x-offset-size, y, x-offset, y)
    c.line(x, y-offset-size, x, y-offset)
    # Bottom-right corner
    c.line(x+w+offset, y, x+w+offset+size, y)
    c.line(x+w, y-offset-size, x+w, y-offset)
    c.restoreState()

def make_sheet(input_pdf, output_pdf, label):
    src = PdfReader(input_pdf)
    xobj = pagexobj(src.pages[0])
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=(PAGE_W, PAGE_H))
    c.setTitle(label)
    xobj_name = makerl(c, xobj)
    c.saveState()
    c.scale(scale_x, scale_y)
    c.doForm(xobj_name)
    c.restoreState()
    for x, y in positions:
        draw_cut_marks(c, x, y, CARD_W_PTS, CARD_H_PTS)
    c.setFillColor(HexColor('#555555'))
    c.setFont('Helvetica', 6)
    c.drawCentredString(PAGE_W/2, 4*mm, f'{label} — 10 cards/sheet — 88.9×50.8mm — ✂ Cortar en marcas')
    c.save()
    with open(output_pdf, 'wb') as f:
        f.write(buf.getvalue())
    print(f'✅ {os.path.basename(output_pdf)}')

configs = [
    ('EkoAI_White_Front.pdf',    'EkoAI_White_Front_cuts.pdf',    'Eko AI — Frente Blanco'),
    ('EkoAI_White_Back.pdf',     'EkoAI_White_Back_cuts.pdf',     'Eko AI — Reverso Blanco'),
    ('BlackVolt_White_Front.pdf', 'BlackVolt_White_Front_cuts.pdf','Black Volt — Frente Blanco'),
    ('BlackVolt_White_Back.pdf',  'BlackVolt_White_Back_cuts.pdf', 'Black Volt — Reverso Blanco'),
]

for inp, out, label in configs:
    make_sheet(f'{CARDS_DIR}/{inp}', f'{CARDS_DIR}/{out}', label)

print('\nDone! All 4 cut-mark PDFs ready.')
