# PyECOM's participant class, an agent participating in a market
# A participant should be comprised of a name, the products it offers, products it demands, and a budget

from .base_item import BaseItem
import numpy as np


class BaseParticipant:

    def __init__(self, name: str,
                 budget: float = np.inf,
                 max_bid: float = np.inf,
                 buy_stock: list[BaseItem] = None,
                 sell_stock: list[BaseItem] = None):

        # Identifier of the participant
        self.name = name

        # Budget is the amount of money that a participant has
        self.budget = budget

        # Max bid is the maximum amount of money that a participant is willing to spend
        self.max_bid = max_bid

        # Buy stock is the list of items that a participant is willing to buy
        self.buy_stock = self.validate_stock(buy_stock)

        # Sell stock is the list of items that a participant is willing to sell
        self.sell_stock = self.validate_stock(sell_stock)

        # Placeholders for buying and selling actions
        self.buy_log = []
        self.sell_log = []

    # Representation of the participant
    def __repr__(self):
        return f'{self.name}'

    @staticmethod
    def merge_stock(stock):

        # Merge items with the same identifier
        merged_stock = []
        for item in stock:
            if item.identifier in [temp.identifier for temp in merged_stock]:
                for merged_item in merged_stock:
                    if merged_item.identifier == item.identifier:
                        merged_item.quantity += item.quantity
            else:
                merged_stock.append(item)

        return merged_stock

    def validate_stock(self, stock):

        # Check if there are no repeated identifier property of the stock
        if len(set([item.identifier for item in stock])) != len(stock):
            return self.merge_stock(stock)

        return stock

    def sell(self, item, quantity, price):

        # Get the item from the sell stock
        temp = [temp for temp in self.sell_stock if temp.identifier == item.identifier]

        # Remove quantity from the item
        temp[0].quantity -= quantity

        # Add to the budget
        self.budget += quantity * price

        # Add to the sell log
        self.sell_log.append(BaseItem(item, price, quantity))

        return

    def buy(self, item, quantity, price):

        # Get the item from the buy stock
        temp = [temp for temp in self.buy_stock if temp.identifier == item.identifier]

        # Remove quantity from the item
        temp[0].quantity -= quantity

        # Remove from the budget
        self.budget -= quantity * price

        # Add to the buy log
        self.buy_log.append(BaseItem(item, price, quantity))

        return

    @staticmethod
    def get_stock_quantity(stock, item) -> int:

        # Get the item from the buy stock
        temp = [temp for temp in stock if temp.identifier == item.identifier]

        # If there is an item, return its quantity
        if len(temp) > 0:
            return temp[0].quantity

        # If there is no item, return 0
        return 0

    @staticmethod
    def get_stock_price(stock, item) -> float:

        # Get the item from the buy stock
        temp = [temp for temp in stock if temp.identifier == item.identifier]

        # If there is an item, return its price
        if len(temp) > 0:
            return temp[0].price

        # If there is no item, return 0
        return 0.0
