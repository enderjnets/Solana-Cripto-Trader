#!/usr/bin/env python3
"""
Convert Markdown files to formatted PDFs with tables and emojis.
"""

import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import re

# Register fonts
try:
    pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    FONT_NAME = 'DejaVu'
    FONT_NAME_BOLD = 'DejaVu-Bold'
except:
    FONT_NAME = 'Helvetica'
    FONT_NAME_BOLD = 'Helvetica-Bold'


def convert_emojis(text):
    """Keep emojis as-is (ReportLab handles them)"""
    return text


def parse_markdown_to_elements(md_content):
    """Parse markdown content and return ReportLab elements"""
    styles = getSampleStyleSheet()
    elements = []

    # Custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontName=FONT_NAME_BOLD,
        fontSize=24,
        textColor=colors.HexColor('#2563EB'),
        spaceAfter=20,
        alignment=TA_CENTER,
    )

    heading1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontName=FONT_NAME_BOLD,
        fontSize=18,
        textColor=colors.HexColor('#1E40AF'),
        spaceBefore=20,
        spaceAfter=10,
    )

    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontName=FONT_NAME_BOLD,
        fontSize=14,
        textColor=colors.HexColor('#3B82F6'),
        spaceBefore=15,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontName=FONT_NAME,
        fontSize=11,
        spaceAfter=8,
        leading=16,
        alignment=TA_JUSTIFY,
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=10,
        textColor=colors.HexColor('#DC2626'),
        backColor=colors.HexColor('#FEF2F2'),
        spaceAfter=8,
    )

    lines = md_content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Title (# with 🎯 or 👋 etc at start)
        if line.startswith('# '):
            text = convert_emojis(line[2:].strip())
            elements.append(Paragraph(text, title_style))
            i += 1
            continue

        # Heading 1 (##)
        elif line.startswith('## '):
            text = convert_emojis(line[3:].strip())
            elements.append(Paragraph(text, heading1_style))
            i += 1
            continue

        # Heading 2 (###)
        elif line.startswith('### '):
            text = convert_emojis(line[4:].strip())
            elements.append(Paragraph(text, heading2_style))
            i += 1
            continue

        # Empty line
        elif not line.strip():
            elements.append(Paragraph('<br/>', body_style))
            i += 1
            continue

        # Horizontal rule (--- or ***)
        elif line.strip().startswith('---') or line.strip().startswith('***'):
            elements.append(PageBreak())
            i += 1
            continue

        # Table starts with |
        elif '|' in line and i + 1 < len(lines) and '|' in lines[i + 1]:
            # Parse table
            table_data = []
            col_widths = []

            # Header row
            header = [cell.strip() for cell in line.split('|')[1:-1]]
            table_data.append([convert_emojis(cell) for cell in header])

            # Separator row
            i += 1
            if not lines[i].strip().startswith('|'):
                # Not a table, treat as regular text
                elements.append(Paragraph(convert_emojis(line), body_style))
                continue

            # Data rows
            i += 1
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [cell.strip() for cell in lines[i].split('|')[1:-1]]
                table_data.append([convert_emojis(cell) for cell in row])
                i += 1

            # Create table
            if table_data:
                num_cols = len(table_data[0])
                col_widths = [A4[0] / num_cols - 100] * num_cols
                table = Table(table_data, colWidths=[A4[0] / num_cols - 50] * num_cols)

                # Table style
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DBEAFE')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1E40AF')),
                    ('FONTNAME', (0, 0), (-1, 0), FONT_NAME_BOLD),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
                ]))

                elements.append(table)
                elements.append(Paragraph('<br/>', body_style))
            continue

        # Code block (starts with ``` or spaces)
        elif line.startswith('```') or line.startswith('    '):
            code_lines = []
            if line.startswith('```'):
                i += 1
                while i < len(lines) and not lines[i].startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                i += 1
            else:
                while i < len(lines) and lines[i].startswith('    '):
                    code_lines.append(lines[i][4:])
                    i += 1

            if code_lines:
                code_text = '<br/>'.join(code_lines)
                elements.append(Paragraph(f'<font name="Courier">{code_text}</font>', code_style))
            continue

        # Regular paragraph
        else:
            text = convert_emojis(line)
            # Convert **bold** and *italic*
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
            elements.append(Paragraph(text, body_style))
            i += 1

    return elements


def markdown_to_pdf(md_file, pdf_file):
    """Convert a markdown file to PDF"""
    md_content = Path(md_file).read_text(encoding='utf-8')

    # Create PDF
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
    )

    elements = parse_markdown_to_elements(md_content)

    # Build PDF
    doc.build(elements)
    print(f"✅ PDF creado: {pdf_file}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python convert_md_to_pdf.py <archivo.md> [archivo_salida.pdf]")
        sys.exit(1)

    md_file = sys.argv[1]

    if len(sys.argv) >= 3:
        pdf_file = sys.argv[2]
    else:
        pdf_file = md_file.replace('.md', '.pdf')

    markdown_to_pdf(md_file, pdf_file)
