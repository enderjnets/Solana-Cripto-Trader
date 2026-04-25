from solana.rpc.api import Client
from solders.pubkey import Pubkey

# Connect to devnet
client = Client('https://api.devnet.solana.com')

# Check wallet balance
wallet = Pubkey.from_string('H9GF6t5hdypfH5PsDhS42sb9ybFWEh5zD5Nht9rQX19a')
response = client.get_balance(wallet)

print(f'Wallet: H9GF6t5hdypfH5PsDhS42sb9ybFWEh5zD5Nht9rQX19a')
print(f'Balance: {response.value} lamports')
print(f'SOL: {response.value / 1e9:.4f}')
