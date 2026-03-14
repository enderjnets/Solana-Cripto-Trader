#!/usr/bin/env python3
"""
Create professional PDF guide for Cristina
Clean design, no ASCII art, proper formatting
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, 
    PageBreak, Spacer, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register fonts
try:
    pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    FONT_NAME = 'DejaVu'
    FONT_NAME_BOLD = 'DejaVu-Bold'
except:
    FONT_NAME = 'Helvetica'
    FONT_NAME_BOLD = 'Helvetica-Bold'


def create_styles():
    """Create custom paragraph styles"""
    styles = getSampleStyleSheet()
    
    # Title
    styles.add(ParagraphStyle(
        name='MainTitle',
        fontName=FONT_NAME_BOLD,
        fontSize=28,
        textColor=colors.HexColor('#1E40AF'),
        alignment=TA_CENTER,
        spaceAfter=30,
        spaceBefore=20,
    ))
    
    # Subtitle
    styles.add(ParagraphStyle(
        name='Subtitle',
        fontName=FONT_NAME,
        fontSize=16,
        textColor=colors.HexColor('#6B7280'),
        alignment=TA_CENTER,
        spaceAfter=40,
    ))
    
    # Section title
    styles.add(ParagraphStyle(
        name='SectionTitle',
        fontName=FONT_NAME_BOLD,
        fontSize=18,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=25,
        spaceAfter=15,
    ))
    
    # Subsection title
    styles.add(ParagraphStyle(
        name='SubsectionTitle',
        fontName=FONT_NAME_BOLD,
        fontSize=14,
        textColor=colors.HexColor('#3B82F6'),
        spaceBefore=15,
        spaceAfter=10,
    ))
    
    # Body text
    styles.add(ParagraphStyle(
        name='CustomBody',
        fontName=FONT_NAME,
        fontSize=11,
        textColor=colors.HexColor('#1F2937'),
        spaceBefore=6,
        spaceAfter=8,
        leading=16,
        alignment=TA_JUSTIFY,
    ))
    
    # Bullet text
    styles.add(ParagraphStyle(
        name='BulletText',
        fontName=FONT_NAME,
        fontSize=11,
        textColor=colors.HexColor('#374151'),
        leftIndent=20,
        spaceBefore=4,
        spaceAfter=4,
        leading=15,
    ))
    
    # Highlight box
    styles.add(ParagraphStyle(
        name='Highlight',
        fontName=FONT_NAME_BOLD,
        fontSize=12,
        textColor=colors.HexColor('#1E40AF'),
        backColor=colors.HexColor('#DBEAFE'),
        spaceBefore=10,
        spaceAfter=10,
        leftIndent=10,
        rightIndent=10,
        borderPadding=10,
    ))
    
    # Conclusion
    styles.add(ParagraphStyle(
        name='Conclusion',
        fontName=FONT_NAME_BOLD,
        fontSize=13,
        textColor=colors.white,
        backColor=colors.HexColor('#059669'),
        spaceBefore=15,
        spaceAfter=15,
        alignment=TA_CENTER,
        borderPadding=15,
    ))
    
    return styles


def create_pdf():
    """Create the professional PDF"""
    pdf_file = "/home/enderj/.openclaw/workspace/docs/Guia_Completa_Agentes_IA.pdf"
    
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=60,
        leftMargin=60,
        topMargin=50,
        bottomMargin=50,
    )
    
    styles = create_styles()
    elements = []
    
    # ===================== PAGE 1: COVER =====================
    elements.append(Spacer(1, 100))
    elements.append(Paragraph("🤖 Agentes de IA", styles['MainTitle']))
    elements.append(Paragraph("Guía Completa", styles['Subtitle']))
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("ChatGPT vs Claude CoWork vs Sistema de Agentes", styles['Subtitle']))
    elements.append(Spacer(1, 50))
    elements.append(Paragraph("Para Cristina - Eko", styles['CustomBody']))
    elements.append(Paragraph("14 de Marzo, 2026", styles['CustomBody']))
    elements.append(PageBreak())
    
    # ===================== PAGE 2: INTRODUCTION =====================
    elements.append(Paragraph("👋 Introducción", styles['SectionTitle']))
    elements.append(Paragraph(
        "Hola Cristina, soy Eko, el asistente de IA de Ender. Él me pidió que te explique "
        "cómo funciona este sistema de agentes que estamos construyendo.",
        styles['CustomBody']
    ))
    
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("💡 La Pregunta Principal", styles['SubsectionTitle']))
    elements.append(Paragraph(
        "<b>¿Cuál es la diferencia entre usar ChatGPT, Claude CoWork y un Sistema de Agentes?</b>",
        styles['CustomBody']
    ))
    
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("🎯 Objetivo de Esta Guía", styles['SubsectionTitle']))
    elements.append(Paragraph(
        "Explicarte de forma clara y sencilla las diferencias entre estas tres tecnologías, "
        "y por qué Ender está invirtiendo en construir un Sistema de Agentes completo.",
        styles['CustomBody']
    ))
    
    elements.append(PageBreak())
    
    # ===================== PAGE 3: CHATGPT =====================
    elements.append(Paragraph("1️⃣ ChatGPT y Chatbots Tradicionales", styles['SectionTitle']))
    
    elements.append(Paragraph("¿Qué es ChatGPT?", styles['SubsectionTitle']))
    elements.append(Paragraph(
        "ChatGPT es un chatbot de OpenAI que responde preguntas y mantiene conversaciones. "
        "Es excelente para obtener información, generar ideas y conversar.",
        styles['CustomBody']
    ))
    
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("✅ Lo que PUEDE hacer:", styles['SubsectionTitle']))
    
    bullets = [
        "Responder preguntas sobre cualquier tema",
        "Generar textos, ideas y contenido",
        "Ayudar con tareas de escritura",
        "Explicar conceptos complejos",
        "Mantener una conversación fluida",
    ]
    for bullet in bullets:
        elements.append(Paragraph(f"• {bullet}", styles['BulletText']))
    
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("❌ Lo que NO puede hacer:", styles['SubsectionTitle']))
    
    bullets = [
        "Ejecutar acciones en el mundo real",
        "Conectarse a APIs externas directamente",
        "Guardar memoria entre sesiones",
        "Trabajar de forma autónoma sin supervisión",
        "Coordinarse con otros sistemas",
    ]
    for bullet in bullets:
        elements.append(Paragraph(f"• {bullet}", styles['BulletText']))
    
    elements.append(Spacer(1, 15))
    elements.append(Paragraph(
        "<b>En resumen:</b> ChatGPT es como un cerebro que puede pensar y hablar, "
        "pero no tiene manos para actuar.",
        styles['CustomBody']
    ))
    
    elements.append(PageBreak())
    
    # ===================== PAGE 4: CLAUDE COWORK =====================
    elements.append(Paragraph("2️⃣ Claude CoWork", styles['SectionTitle']))
    
    elements.append(Paragraph("¿Qué es Claude CoWork?", styles['SubsectionTitle']))
    elements.append(Paragraph(
        "Claude CoWork es una función de Anthropic que permite que varias instancias de Claude "
        "trabajen juntas en un chat grupal. Es como tener varios Claudes conversando entre sí.",
        styles['CustomBody']
    ))
    
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("✅ Ventajas:", styles['SubsectionTitle']))
    
    bullets = [
        "Múltiples Claudes colaboran entre sí",
        "Diferentes perspectivas sobre un mismo problema",
        "Excelente para brainstorming y análisis",
        "Fácil de usar dentro de la plataforma de Anthropic",
    ]
    for bullet in bullets:
        elements.append(Paragraph(f"• {bullet}", styles['BulletText']))
    
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("⚠️ Limitaciones:", styles['SubsectionTitle']))
    
    bullets = [
        "Solo puedes usar Claude (un solo proveedor)",
        "Herramientas limitadas a lo que Anthropic ofrece",
        "No puedes cambiar la arquitectura del sistema",
        "Memoria limitada a la sesión actual",
        "No se conecta a sistemas externos (YouTube, email, APIs)",
        "No ejecuta acciones reales en el mundo",
    ]
    for bullet in bullets:
        elements.append(Paragraph(f"• {bullet}", styles['BulletText']))
    
    elements.append(Spacer(1, 15))
    elements.append(Paragraph(
        "<b>En resumen:</b> Claude CoWork es un chat grupal de Claudes. Es mejor que ChatGPT "
        "para colaboración, pero sigue sin poder ejecutar acciones reales.",
        styles['CustomBody']
    ))
    
    elements.append(PageBreak())
    
    # ===================== PAGE 5: SYSTEM OF AGENTS =====================
    elements.append(Paragraph("3️⃣ Sistema de Agentes (Lo que Ender está construyendo)", styles['SectionTitle']))
    
    elements.append(Paragraph("¿Qué es un Sistema de Agentes?", styles['SubsectionTitle']))
    elements.append(Paragraph(
        "Un Sistema de Agentes es una arquitectura propia donde cada agente tiene roles específicos, "
        "herramientas personalizadas y puede conectarse con sistemas externos. Es como construir "
        "una empresa digital completa.",
        styles['CustomBody']
    ))
    
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("🚀 Características Principales:", styles['SubsectionTitle']))
    
    bullets = [
        "<b>Multi-proveedor:</b> Usa Claude, GLM-4.7, MiniMax, Qwen, y cualquier otro modelo",
        "<b>Herramientas ilimitadas:</b> GitHub, YouTube, Email, Calendar, Navegador, APIs, etc.",
        "<b>Arquitectura propia:</b> Tú decides cómo funciona el sistema",
        "<b>Memoria persistente:</b> Bases de datos, archivos, historiales",
        "<b>Integración real:</b> Se conecta con el mundo real",
        "<b>Fallback automático:</b> Si un modelo falla, usa otro",
        "<b>Independencia:</b> No dependes de un solo proveedor",
    ]
    for bullet in bullets:
        elements.append(Paragraph(f"• {bullet}", styles['BulletText']))
    
    elements.append(Spacer(1, 15))
    elements.append(Paragraph(
        "<b>En resumen:</b> Un Sistema de Agentes es como tener una empresa completa con "
        "empleados expertos que trabajan 24/7 y pueden ejecutar acciones reales.",
        styles['CustomBody']
    ))
    
    elements.append(PageBreak())
    
    # ===================== PAGE 6: COMPARISON TABLE =====================
    elements.append(Paragraph("📊 Comparación Completa", styles['SectionTitle']))
    
    # Create comparison table
    table_data = [
        ['Característica', 'ChatGPT', 'Claude CoWork', 'Sistema de Agentes'],
        ['Proveedores', 'Solo OpenAI', 'Solo Anthropic', 'Multi-proveedor'],
        ['Herramientas externas', 'Limitadas', 'Limitadas', 'Ilimitadas'],
        ['Memoria persistente', 'No', 'No', 'Sí'],
        ['Ejecutar acciones', 'No', 'No', 'Sí'],
        ['Integración APIs', 'Básica', 'Básica', 'Avanzada'],
        ['Independencia', 'No', 'No', 'Sí'],
        ['Costo optimizado', 'No', 'No', 'Sí'],
        ['Escalabilidad', 'Limitada', 'Limitada', 'Ilimitada'],
        ['Autonomía', 'Conversacional', 'Conversacional', 'Ejecutiva'],
    ]
    
    table = Table(table_data, colWidths=[100, 90, 100, 120])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E40AF')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), FONT_NAME_BOLD),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTNAME', (0, 1), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F3F4F6')]),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph(
        "<b>Conclusión:</b> ChatGPT y Claude CoWork son excelentes para conversaciones, "
        "pero un Sistema de Agentes es necesario cuando quieres automatización real.",
        styles['CustomBody']
    ))
    
    elements.append(PageBreak())
    
    # ===================== PAGE 7: EXAMPLE - BITTRADER =====================
    elements.append(Paragraph("🎬 Ejemplo Real: BitTrader (Canal de YouTube)", styles['SectionTitle']))
    
    elements.append(Paragraph(
        "Ender tiene un canal de YouTube llamado BitTrader sobre trading y criptomonedas. "
        "Este es un ejemplo de cómo los agentes trabajan automáticamente:",
        styles['CustomBody']
    ))
    
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Proceso Completo (sin intervención manual):", styles['SubsectionTitle']))
    
    steps = [
        ("<b>Paso 1 - Investigación:</b> El agente Scout navega la web, busca noticias de criptomonedas, "
         "analiza tendencias de Twitter y detecta temas relevantes."),
        ("<b>Paso 2 - Decisión:</b> El CEO Agent analiza los datos y decide qué tema será el video."),
        ("<b>Paso 3 - Creación:</b> El Creator Agent investiga el tema y escribe un guión completo "
         "con hook, problema, solución, ejemplos y llamada a la acción."),
        ("<b>Paso 4 - Optimización:</b> El MrBeast Agent optimiza el título y genera un thumbnail "
         "usando las estrategias del youtuber más exitoso del mundo."),
        ("<b>Paso 5 - Producción:</b> El Producer Agent genera el audio, crea el visual, añade subtítulos, "
         "valida calidad y genera el video final."),
        ("<b>Paso 6 - Publicación:</b> El Queue Processor sube el video automáticamente a YouTube "
         "a la hora programada."),
    ]
    
    for step in steps:
        elements.append(Paragraph(f"• {step}", styles['BulletText']))
        elements.append(Spacer(1, 5))
    
    elements.append(Spacer(1, 15))
    elements.append(Paragraph(
        "<b>Tiempo total:</b> 15-20 minutos (automático) vs. días de trabajo manual",
        styles['CustomBody']
    ))
    
    elements.append(PageBreak())
    
    # ===================== PAGE 8: EXAMPLE - ECO =====================
    elements.append(Paragraph("🏢 Ejemplo Real: Eco (Automatización para Negocios)", styles['SectionTitle']))
    
    elements.append(Paragraph(
        "Ender está creando un negocio llamado Eco que ofrece servicios de automatización con IA "
        "para empresas. Este es otro ejemplo de cómo trabajan los agentes:",
        styles['CustomBody']
    ))
    
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Proceso de Adquisición de Clientes:", styles['SubsectionTitle']))
    
    steps = [
        ("<b>Investigación:</b> El agente Scout busca negocios locales en Denver que podrían "
         "beneficiarse de automatización (salones, restaurantes, retail)."),
        ("<b>Análisis:</b> El Marketing Agent analiza cada negocio, identifica problemas que "
         "pueden ser automatizados y calcula el ROI potencial."),
        ("<b>Propuesta:</b> El sistema genera una propuesta personalizada para cada cliente potencial, "
         "incluyendo servicios específicos y pricing."),
        ("<b>Contacto:</b> El sistema puede enviar emails automáticamente o incluso hacer llamadas "
         "telefónicas usando VAPI."),
        ("<b>Seguimiento:</b> El sistema hace seguimiento automático de cada lead, programa "
         "reuniones en el calendario y envía recordatorios."),
    ]
    
    for step in steps:
        elements.append(Paragraph(f"• {step}", styles['BulletText']))
        elements.append(Spacer(1, 5))
    
    elements.append(Spacer(1, 15))
    elements.append(Paragraph(
        "<b>Resultado:</b> Un proceso completo de ventas automatizado que trabaja 24/7.",
        styles['CustomBody']
    ))
    
    elements.append(PageBreak())
    
    # ===================== PAGE 9: AGENTS STRUCTURE =====================
    elements.append(Paragraph("👥 Los Agentes de Ender", styles['SectionTitle']))
    
    elements.append(Paragraph(
        "Ender tiene varios agentes especializados trabajando en sus proyectos:",
        styles['CustomBody']
    ))
    
    # Agents table
    agents_data = [
        ['Agente', 'Modelo', 'Función'],
        ['CEO Agent', 'Claude Opus 4.6', 'Coordina todo, detecta problemas, toma decisiones'],
        ['Ingeniero', 'Claude Opus 4.6', 'Escribe código, arregla bugs, crea features'],
        ['Marketing', 'Claude Sonnet 4.6', 'Investiga mercado, encuentra clientes'],
        ['MrBeast', 'Claude Sonnet 4.6', 'Optimiza títulos y thumbnails virales'],
        ['Creator', 'Claude Sonnet 4.6', 'Genera guiones de videos'],
        ['Producer', 'Claude Sonnet 4.6', 'Produce videos completos'],
        ['Scout', 'Claude Sonnet 4.6', 'Investiga tendencias y noticias'],
    ]
    
    agents_table = Table(agents_data, colWidths=[100, 120, 200])
    agents_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), FONT_NAME_BOLD),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTNAME', (0, 1), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#EFF6FF')]),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    
    elements.append(agents_table)
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("🧠 ¿Por qué diferentes modelos?", styles['SubsectionTitle']))
    elements.append(Paragraph(
        "<b>Opus 4.6:</b> Para tareas complejas de razonamiento (CEO, Ingeniero)",
        styles['CustomBody']
    ))
    elements.append(Paragraph(
        "<b>Sonnet 4.6:</b> Para tareas creativas y de marketing (resto de agentes)",
        styles['CustomBody']
    ))
    elements.append(Paragraph(
        "Esto optimiza costos: usas el modelo más caro solo donde es necesario.",
        styles['CustomBody']
    ))
    
    elements.append(PageBreak())
    
    # ===================== PAGE 10: BENEFITS =====================
    elements.append(Paragraph("✨ Beneficios del Sistema de Agentes", styles['SectionTitle']))
    
    benefits = [
        ("<b>1. Automatización completa:</b> Tareas complejas se ejecutan sin intervención manual"),
        ("<b>2. Velocidad:</b> Lo que tomaba días, ahora toma minutos"),
        ("<b>3. Escalabilidad:</b> Puedes agregar nuevos agentes cuando quieras"),
        ("<b>4. Optimización de costos:</b> Usas el modelo adecuado para cada tarea"),
        ("<b>5. Independencia:</b> No dependes de un solo proveedor"),
        ("<b>6. Memoria persistente:</b> El sistema recuerda todo"),
        ("<b>7. Integración real:</b> Conexión con el mundo real (YouTube, email, APIs, etc.)"),
        ("<b>8. Trabajo 24/7:</b> Los agentes trabajan todo el día sin descanso"),
    ]
    
    for benefit in benefits:
        elements.append(Paragraph(f"• {benefit}", styles['BulletText']))
        elements.append(Spacer(1, 5))
    
    elements.append(PageBreak())
    
    # ===================== PAGE 11: CONCLUSION =====================
    elements.append(Paragraph("🎯 Conclusión Final", styles['SectionTitle']))
    
    elements.append(Spacer(1, 20))
    
    # Final comparison table
    final_data = [
        ['Tecnología', 'Puede hacer'],
        ['ChatGPT', 'Responder preguntas'],
        ['Claude CoWork', 'Varios Claudes colaboran en conversaciones'],
        ['Sistema de Agentes', 'Ejecutar tareas reales, construir sistemas completos'],
    ]
    
    final_table = Table(final_data, colWidths=[150, 280])
    final_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#059669')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME_BOLD),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#059669')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#ECFDF5')),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    
    elements.append(final_table)
    elements.append(Spacer(1, 30))
    
    elements.append(Paragraph(
        "<b>La diferencia clave:</b>",
        styles['CustomBody']
    ))
    
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "ChatGPT y Claude CoWork son herramientas de conversación. "
        "Un Sistema de Agentes es una empresa digital completa que trabaja por ti.",
        styles['CustomBody']
    ))
    
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "Ender está construyendo sistemas que pueden ejecutar acciones reales, "
        "no solo conversar. Esto es lo que lo diferencia y lo que hace que su proyecto sea tan potente.",
        styles['CustomBody']
    ))
    
    elements.append(Spacer(1, 40))
    elements.append(Paragraph(
        "🤓 ¿Tienes más preguntas? Ender puede explicarte más detalles.",
        styles['CustomBody']
    ))
    
    # Build PDF
    doc.build(elements)
    print(f"✅ PDF profesional creado: {pdf_file}")
    return pdf_file


if __name__ == '__main__':
    create_pdf()
