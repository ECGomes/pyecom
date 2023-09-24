# PyECOM's base market class
# A market should be comprised of a set of agents, a set of products.
#
# A market should be able to:
# - Get the market data
# - Get the market
# - Get the market's agents
# - Get the market's products
# - Get the market's transactions
# - Get the market's transactions' prices
# - Get the market's transactions' quantities
# - Get the market's transactions' agents
# - Get the market's transactions' products
# - Get the market's transactions' timestamps

from .base_participant import BaseParticipant
from .base_item import BaseItem


class BaseMarket:

    def __init__(self, participants: list[BaseParticipant]):
        self.participants = participants
        self.transaction_history = []
        pass

    def create_transaction(self,
                           buyer: BaseParticipant,
                           seller: BaseParticipant,
                           item: BaseItem,
                           quantity: float,
                           price: float):

        # Create a transaction
        transaction = {
            'buyer': buyer,
            'seller': seller,
            'item': item,
            'quantity': quantity,
            'price': price
        }

        # Add the transaction to the transaction history
        self.transaction_history.append(transaction)

        # Add the transaction to the buyer's and seller's logs
        buyer.buy_log.append(transaction)
        seller.sell_log.append(transaction)
