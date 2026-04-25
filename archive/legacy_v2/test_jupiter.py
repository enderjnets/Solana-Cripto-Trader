#!/usr/bin/env python3
"""
Jupiter Ultra API Quick Test
"""

import asyncio
from api.api_integrations import JupiterClient, SOL, USDC


async def main():
    print("="*60)
    print("🚀 JUPITER API QUICK TEST")
    print("="*60)
    
    wallet = "65YqSYGwR6UNCUmeaKt1V1HV99Ky1tii2bgg6jwJSGN3"
    client = JupiterClient()
    
    try:
        # 1. Price
        print("\n📊 PRICE API")
        price = await client.get_token_price(SOL)
        print(f"  SOL: ${price:.2f}")
        
        # 2. Quote
        print("\n💱 QUOTE: 1 SOL → USDC")
        order = await client.quote_sol_to_usdc(1.0)
        out = client.micro_to_usdc(int(order.out_amount))
        print(f"  → {out:.2f} USDC | Impact: {order.price_impact_pct}%")
        print(f"  Route: {len(order.route_plan)} hops")
        print(f"  Request ID: {order.request_id[:20]}...")
        
        # 3. Quote with wallet
        print(f"\n🔐 QUOTE WITH WALLET")
        order_tx = await client.quote_sol_to_usdc(1.0, wallet)
        has_tx = order_tx.transaction is not None
        print(f"  Transaction: {'✅' if has_tx else '❌'}")
        
        # 4. Holdings
        print(f"\n💰 HOLDINGS")
        holdings = await client.get_holdings(wallet)
        print(f"  Keys: {list(holdings.keys())}")
        
        # 5. Search
        print("\n🔍 SEARCH 'JUP'")
        tokens = await client.search_tokens("JUP")
        print(f"  Found: {len(tokens)} tokens")
        for t in tokens[:3]:
            print(f"    {t.get('symbol')}: {t.get('name')}")
        
        # 6. Trending
        print("\n📈 TRENDING (1h)")
        trending = await client.get_trending_tokens("1h")
        for t in trending[:3]:
            print(f"    {t.get('symbol')}: {t.get('name')}")
        
    finally:
        await client.close()
    
    print("\n" + "="*60)
    print("✅ ALL WORKING - https://dev.jup.ag/")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
