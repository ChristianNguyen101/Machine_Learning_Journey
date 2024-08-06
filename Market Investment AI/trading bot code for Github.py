#trading bot for Github

#intial dependencies
from lumibot.brokers import Alpaca  
#Used for Backtesting validation if not running in Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
#Standardizes date and time formats
from datetime import datetime
#Used to collect news
from alpaca_trade_api import REST
#Used to find time difference
from timedelta import Timedelta
#Used to adjust trading strategies based on neural network code 
from finbertutils import estimate_sentiment

#User Credentials 
#Input your own credentials
API_KEY = ""
API_SECRET = ""
BASE_URL = ""

ALPACA_CREDS ={
    #calls the API key inputted
    "API_KEY": API_KEY,
    #calls the secret API key inputted
    "API_SECRET": API_SECRET,
    # This means that no real money is traded (SIMULATION)
    "PAPER": True
    
}


class MLTrader(Strategy):
    def initialize(self, symbol: str = "SPY", cash_at_risk: float = 0.25):
        self.symbol = symbol
        # Sets Training Schedule
        self.sleeptime = "24H"
        # Sets Baseline 
        self.last_trade = None
        #Cash at risk is how much money willing to be traded
        self.cash_at_risk = cash_at_risk
        #Connects the Trading Strategy with Alpaca
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self):
        # Determines how much cash in balance from Alpaca
        cash = self.get_cash()
        # Determines the Last Ticker Price of the Stock/ETF purchased
        last_price = self.get_last_price(self.symbol)
        # Determines how many units to be purchased
        quantity = round(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity

    def get_dates(self):
        # Determines End Date
        today = self.get_datetime()
        #Determines Start Date
        three_days_prior = today - Timedelta(days=3)
        # Reformats data for the Alpaca API 
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        # Collects News from relevant timepoints
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        # Utilizes the Estimate Sentiment functionality from the Finber_utlils file
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()
        # Makes Sure we don't buy what I don't have
        if cash > last_price:
            # If bullish sentiment
            if sentiment == "positive" and probability > .999:
                # Removes Any Floating Sell Orders 
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.3,
                    stop_loss_price=last_price * .85
                )
                self.submit_order(order)
                self.last_trade = "buy"
            #elif bearish sentiment
            elif sentiment == "negative" and probability > .999:
                # Removes Any Floating Buy Orders 
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * .75,
                    stop_loss_price=last_price * 1.025
                )
                self.submit_order(order)
                self.last_trade = "sell"

# Validation starts at the beginning of December 2023 and ends December 2023
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

# Defines the Account
broker = Alpaca(ALPACA_CREDS)

# Defines our trading strategy
strategy = MLTrader(name='machine_learning_strategy',
                    broker=broker,
                    parameters={"symbol": "SPY", "cash_at_risk": .25})

# Defines our backtesting (VALIDATION)
#backtest_results = strategy.backtest(
    #YahooDataBacktesting,
    #start_date,
    #end_date,
    #parameters={"symbol": "SPY", "cash_at_risk": .25}
#)
# Connects our ML trading Strategy to the ALPACA Paper Trade API
trader = Trader()
trader.add_strategy(strategy)
trader.run_all()
