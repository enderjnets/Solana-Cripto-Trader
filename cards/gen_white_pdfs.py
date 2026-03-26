from playwright.sync_api import sync_playwright
import os

CARDS_DIR = '/home/enderj/.openclaw/workspace/cards'

files = [
    ('sheet_eko_front_white.html', 'EkoAI_White_Front.pdf'),
    ('sheet_eko_back_white.html',  'EkoAI_White_Back.pdf'),
    ('sheet_bv_front_white.html',  'BlackVolt_White_Front.pdf'),
    ('sheet_bv_back_white.html',   'BlackVolt_White_Back.pdf'),
]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    for html_file, pdf_file in files:
        html_path = f'file://{CARDS_DIR}/{html_file}'
        page.goto(html_path)
        page.wait_for_timeout(1500)
        page.pdf(
            path=f'{CARDS_DIR}/{pdf_file}',
            width='210mm',
            height='297mm',
            print_background=True,
            margin={'top': '0', 'bottom': '0', 'left': '0', 'right': '0'}
        )
        print(f'✅ {pdf_file}')
    browser.close()

print('All base PDFs generated.')
