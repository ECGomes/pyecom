# Implements a transaction system for the market

from .base_participant import BaseParticipant
from .base_item import BaseItem


class BaseTransaction:

    def __init__(self,
                 buyer: BaseParticipant,
                 seller: BaseParticipant,
                 item: BaseItem,
                 quantity: float,
                 price: float):
        self.buyer = buyer
        self.seller = seller
        self.item = item
        self.quantity = quantity
        self.price = price
        self.verified = False
        self.verified_timestamp = None
        self.executed = False
        self.executed_timestamp = None
        self.completed = False
        self.completed_timestamp = None

    def __repr__(self):
        return f'{self.buyer.name} bought {self.quantity} ' \
               f'of {self.item.identifier} from {self.seller.name} ' \
               f'for {self.price * self.quantity} ' \
               f'at {self.price} per unit'


class BaseTransactionSystem:

    def __init__(self, limit_buy: bool = True, limit_sell: bool = True):
        # Limit buy and sell transactions in cases where the buyer or seller does not have enough
        # budget or stock
        self.limit_buy = limit_buy
        self.limit_sell = limit_sell

        # Create lists for the transactions
        self.created_transactions = []
        self.verified_transactions = []
        self.executed_transactions = []
        self.completed_transactions = []
        self.incomplete_transactions = []

    def create_transaction(self,
                           buyer: BaseParticipant,
                           seller: BaseParticipant,
                           item: BaseItem,
                           quantity: float,
                           price: float):
        new_transaction = BaseTransaction(buyer, seller, item, quantity, price)

        self.created_transactions.append(new_transaction)

        return new_transaction

    def verify_transaction(self,
                           transaction: BaseTransaction,
                           timestamp) -> BaseTransaction:

        # Check if the transaction is valid
        # Check the seller for stock
        # Get the seller stock
        seller_stock = transaction.seller.get_stock_quantity(stock=transaction.seller.sell_stock,
                                                             item=transaction.item)

        # Check if the seller has enough stock
        flag_seller = seller_stock >= transaction.quantity
        if not flag_seller and self.limit_sell:
            # If the seller does not have enough stock and the limit sell flag is set to true,
            # sell what is possible
            transaction.quantity = seller_stock

        # Check if the buyer has enough budget
        flag_buyer = transaction.buyer.budget >= transaction.price * transaction.quantity
        if not flag_buyer and self.limit_buy:
            # If the buyer does not have enough budget and the limit buy flag is set to true,
            # buy what is possible
            transaction.quantity = transaction.buyer.budget / transaction.price

        # Check if the transaction is valid
        transaction.verified = flag_seller and flag_buyer
        transaction.verified_timestamp = timestamp

        # Add the transaction to the verified transactions list
        self.verified_transactions.append(transaction)

        return transaction

    def execute_transaction(self, transaction: BaseTransaction, timestamp: float) -> BaseTransaction:

        # Check if the transaction is valid
        if not transaction.verified:
            return transaction

        # Execute the transaction
        # As everything should OK by now, we simply remove the stock from the seller and add it to
        # the buyer
        transaction.seller.sell(transaction.item, transaction.quantity, transaction.price)
        transaction.buyer.buy(transaction.item, transaction.quantity, transaction.price)

        # Flag the transaction as executed
        transaction.executed = True
        transaction.executed_timestamp = timestamp

        # Add the transaction to the executed transactions list
        self.executed_transactions.append(transaction)

        return transaction

    def execute(self,
                buyer: BaseParticipant,
                seller: BaseParticipant,
                item: BaseItem,
                quantity: int,
                price: float, timestamp) -> None:

        transaction = self.create_transaction(buyer=buyer,
                                              seller=seller,
                                              item=item,
                                              quantity=quantity,
                                              price=price)

        transaction = self.verify_transaction(transaction=transaction, timestamp=timestamp)

        transaction = self.execute_transaction(transaction=transaction, timestamp=timestamp)

        print(transaction)

        # Check if the transaction was executed
        if transaction.executed:
            # If the transaction was executed, append it to the completed transactions list
            self.completed_transactions.append(transaction)
        else:
            # If the transaction was not executed, append it to the incomplete transactions list
            self.incomplete_transactions.append(transaction)

        return
