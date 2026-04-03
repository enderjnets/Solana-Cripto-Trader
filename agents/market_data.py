# market_data.py - Wrapper para MarketScannerAgent
# El orchestrator espera md.run() que retorna {"prices_ok": N}
import logging
from market_scanner_agent import MarketScannerAgent

log = logging.getLogger(__name__)

_scanner = None

def run(debug=False):
    """Wrapper que el orchestrator espera: md.run(debug=debug)
    Retorna dict con prices_ok = número de precios obtenidos.
    """
    global _scanner
    if _scanner is None:
        _scanner = MarketScannerAgent()
    
    try:
        opportunities = _scanner.scan_market()
        return {"prices_ok": len(opportunities), "opportunities": opportunities}
    except Exception as e:
        log.error(f"market_data.run() failed: {e}")
        return {"prices_ok": 0, "error": str(e)}
