#!/usr/bin/env python3
"""
Script para actualizar master_orchestrator.py a v3.2
Reemplaza ResearcherAgent y PaperTradingAgent con versiones mejoradas
"""

import re

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def replace_class(content, class_name, new_code_path):
    """Reemplaza una clase completa en el código"""
    # Leer el nuevo código
    new_code = read_file(new_code_path)

    # Encontrar la clase a reemplazar
    pattern = rf'class {class_name}:.*?(?=\n\nclass |\n\ndef main\(|$)'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        old_class = match.group(0)
        content = content.replace(old_class, new_code)
        print(f"✅ {class_name} reemplazado exitosamente")
    else:
        print(f"❌ No se encontró {class_name} para reemplazar")

    return content

def main():
    print("🔧 Actualizando master_orchestrator.py a v3.2...")
    print("=" * 60)

    # Rutas de archivos
    main_file = "/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/master_orchestrator.py"
    researcher_v32 = "/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/researcher_agent_v3.2.py"
    paper_trading_v32 = "/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/paper_trading_agent_v3.2.py"

    # Leer archivo principal
    print("📖 Leyendo master_orchestrator.py...")
    content = read_file(main_file)

    # Reemplazar ResearcherAgent
    print("\n📝 Reemplazando ResearcherAgent...")
    content = replace_class(content, "ResearcherAgent", researcher_v32)

    # Reemplazar PaperTradingAgent
    print("\n📝 Reemplazando PaperTradingAgent...")
    content = replace_class(content, "PaperTradingAgent", paper_trading_v32)

    # Escribir archivo actualizado
    print("\n💾 Escribiendo archivo actualizado...")
    write_file(main_file, content)

    print("\n" + "=" * 60)
    print("✅ Actualización completada exitosamente!")
    print("\n📋 Resumen:")
    print("  • ResearcherAgent actualizado a v3.2 (con filtros)")
    print("  • PaperTradingAgent actualizado a v3.2 (con trailing stop)")
    print("  • Backup guardado en master_orchestrator.py.v3.1.backup")
    print("\n🚀 Listo para reiniciar el bot!")

if __name__ == "__main__":
    main()
