# Base market class
# A market should be comprised of:
# - A set of agents
# - A set of products, associated to a set of agents
# - A transaction system
# - A price system (can be in the form of an auction)

from .base_participant import BaseParticipant
from .base_transaction import BaseTransactionSystem
from .base_pricing import BasePricingSystem
from .base_item import BaseItem


class BaseMarket:

    def __init__(self, participants: list[BaseParticipant]):
        """
        Base market class
        :type participants: list[BaseParticipant]
        :param participants: List of participants in the market
        """
        self.participants = participants
        self.transaction_system = None
        self.pricing_system = None
        pass

    def __setattr__(self, key, value):
        """
        Prevents the user from changing the transaction system
        :param key:
        :param value:
        """
        if key == 'transaction_system':
            if value is not None:
                return
        super().__setattr__(key, value)

    def set_transaction_system(self, transaction_system: BaseTransactionSystem):
        """
        Set the transaction system of the market
        :type transaction_system: BaseTransactionSystem
        :param transaction_system: Transaction system to be used
        :return: None
        """
        self.transaction_system = transaction_system

    def set_pricing_system(self, pricing_system: BasePricingSystem):
        """
        Set the pricing system of the market
        :type pricing_system: BasePricingSystem
        :param pricing_system: Pricing system to be used
        :return: None
        """
        self.pricing_system = pricing_system

    def get_sellers(self, item: BaseItem):
        """
        Get the sellers of a given item
        :type item: BaseItem
        :param item: Item to be sold
        :return: Sellers of a given item
        """
        sellers = []
        for participant in self.participants:
            if item.identifier in [temp.identifier for temp in participant.sell_stock]:
                sellers.append(participant)
        return sellers

    def get_buyers(self, item: BaseItem):
        """
        Get the buyers of a given item
        :type item: BaseItem
        :param item: Item to be bought
        :return:
        """
        buyers = []
        for participant in self.participants:
            if item.identifier in [temp.identifier for temp in participant.buy_stock]:
                buyers.append(participant)
        return buyers

    def iterate(self, item: BaseItem):
        """
        Solve the market
        :param item: Item to be sold
        :return: None
        """
        # Get the sellers and buyers
        sellers = self.get_sellers(item)
        buyers = self.get_buyers(item)

        # Iterate while there are buyers
        while len(buyers) > 0:
            # Solve the market
            buyer, seller, item, quantity, price = self.pricing_system.solve(buyers, sellers, item)

            # If there is no buyer, return
            if buyer is None:
                return

            # If there is no seller, return
            if seller is None:
                return

            # If the quantity is 0, return
            if quantity == 0:
                return

            # Create a transaction based on the buyer, seller, item, quantity and price
            self.transaction_system.execute(buyer, seller, item, quantity, price)

            # Check again the buyers and sellers
            sellers = self.get_sellers(item)
            buyers = self.get_buyers(item)
