import os
import asyncio
import logging
import aiohttp
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        # Environment variables (will be set in Render)
        self.api_key = os.getenv('BINANCE_API_KEY', 'kPlI3h35AjdDKZaiZGp8icxtCBpygugH0WJ4kPJueVWXK3qkF1Dv2sahbUl3YGdq')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY', 'sWsm8qpcBvaHWBkGJRVGOVgSy72s7HRXYctMvgNOLMPSXxxASFYmoDCV4ClnGU5b')
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '7703380978:AAH87J_X5KNNihSCPdX0D-zeJng-L5hqKYo')
        
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        self.chat_ids = set()
        self.balance = 10000.0
        
    async def get_price(self, symbol):
        try:
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'symbol': symbol,
                            'price': float(data['lastPrice']),
                            'change': float(data['priceChangePercent'])
                        }
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
        return None
    
    async def detect_opportunity(self, data):
        if not data:
            return None
            
        if data['change'] < -5:  # 5% drop = crash opportunity
            return f"🚨 CRASH ALERT: {data['symbol']} dropped {data['change']:.2f}% to ${data['price']:.4f}"
        elif data['change'] > 8:  # 8% pump
            return f"📈 PUMP ALERT: {data['symbol']} pumped +{data['change']:.2f}% to ${data['price']:.4f}"
        return None
    
    async def get_telegram_updates(self):
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        for update in data.get('result', []):
                            if 'message' in update:
                                chat_id = update['message']['chat']['id']
                                self.chat_ids.add(chat_id)
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    async def send_alert(self, message):
        for chat_id in self.chat_ids:
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                data = {'chat_id': chat_id, 'text': message}
                async with aiohttp.ClientSession() as session:
                    await session.post(url, json=data)
            except Exception as e:
                logger.error(f"Send error: {e}")
    
    async def scan_markets(self):
        logger.info("Scanning markets...")
        
        opportunities = []
        for symbol in self.symbols:
            data = await self.get_price(symbol)
            if data:
                logger.info(f"{symbol}: ${data['price']:.4f} ({data['change']:+.2f}%)")
                
                alert = await self.detect_opportunity(data)
                if alert:
                    opportunities.append(alert)
        
        if opportunities:
            for alert in opportunities:
                await self.send_alert(alert)
                logger.info(f"Sent alert: {alert}")
        else:
            logger.info("No opportunities found")
    
    async def run(self):
        logger.info("🚀 Trading Bot Started on Render!")
        
        # Send startup message
        await asyncio.sleep(2)
        await self.get_telegram_updates()
        if self.chat_ids:
            await self.send_alert("🤖 AI Trading Bot is now running 24/7 on Render!\nScanning for crash opportunities every 60 seconds.")
        
        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"--- SCAN CYCLE #{cycle} ---")
                
                # Check for new Telegram users
                await self.get_telegram_updates()
                
                # Scan markets
                await self.scan_markets()
                
                # Status update every 10 cycles
                if cycle % 10 == 0:
                    status = f"✅ Status Update #{cycle//10}\nTime: {datetime.now().strftime('%H:%M:%S')}\nBalance: ${self.balance:,.2f}\nActive users: {len(self.chat_ids)}"
                    await self.send_alert(status)
                
                logger.info("Waiting 60 seconds...")
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(30)

if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.run())
