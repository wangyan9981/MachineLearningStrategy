from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import threading

class TradingApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        
    def nextValidId(self, orderId):
        self.order_id = orderId
        print("Connected to IBKR. Ready to trade.")

def execute_trade(symbol, action, quantity):
    """
    Sends market orders to IBKR (paper trading account).
    """
    app = TradingApp()
    app.connect("127.0.0.1", 7497, clientId=1)
    thread = threading.Thread(target=app.run)
    thread.start()
    
    order = Order()
    order.action = action  # "BUY" or "SELL"
    order.orderType = "MKT"
    order.totalQuantity = quantity
    app.placeOrder(app.order_id, Contract(symbol=symbol), order)