# Base pricing mechanisms
from typing import Union

from .base_participant import BaseParticipant
from .base_item import BaseItem


class BasePricingSystem:

    def __init__(self, baseline_price: float = 0.0):

        # Minimum price of the item to be sold
        self.baseline_price = baseline_price
        pass

    @staticmethod
    def get_lowest_seller(sellers: list[BaseParticipant],
                          item: BaseItem) -> Union[BaseParticipant, None]:
        """
        Get the lowest price of an item
        :param sellers: Seller list
        :param item: Item to be sold
        :return: BaseParticipant or None
        """

        item_sellers = [seller for seller in sellers if seller.get_stock_quantity(seller.sell_stock,
                                                                                  item) > 0]

        # If there are no sellers, return None
        if len(item_sellers) == 0:
            return None
        # We don't need to sort if there is only one seller
        elif len(item_sellers) == 1:
            return item_sellers[0]

        # Sort the sellers by price
        item_sellers.sort(key=lambda x: x.get_stock_quantity(x.sell_stock, item))

        # Return the lowest price
        return item_sellers[0]

    def solve(self,
              buyers: list[BaseParticipant],
              sellers: list[BaseParticipant],
              item: BaseItem) -> Union[tuple[BaseParticipant, BaseParticipant, BaseItem, int, float], None]:
        """
        Solve the auction
        :param buyers:
        :param sellers:
        :param item:
        :return: Tuple of (buyer, seller, item, quantity, price)
        """
        raise NotImplementedError


class BaseAuction(BasePricingSystem):

    def __init__(self, price_increment: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.price_increment = price_increment

    def bid(self,
            buyers: list[BaseParticipant],
            sellers: list[BaseParticipant],
            item: BaseItem):
        """
        Bid function. Checks if the bid is valid and returns the winner
        :param buyers: List of buyers
        :param sellers: List of sellers
        :param item: Item to be auctioned
        :return: Winner of the auction
        """

        lowest_seller = self.get_lowest_seller(sellers, item)
        current_price = None

        # Check the price of the lowest seller
        if lowest_seller is not None:
            current_price = lowest_seller.get_stock_price(lowest_seller.sell_stock, item)
        else:
            # If there are no sellers, set the lowest price to the baseline price
            # Equivalent to buying from the grid
            current_price = self.baseline_price

        # Get the buyers that want to get the item
        possible_buyers = [buyer for buyer in buyers if buyer.get_stock_quantity(buyer.buy_stock, item) > 0]

        # Iterate while there are buyers
        while len(possible_buyers) > 0:

            # Get the buyers that can pay the price
            possible_buyers = [buyer for buyer in possible_buyers if buyer.budget >= current_price]

            # If there is more than, increase the price
            if len(possible_buyers) > 1:
                current_price += self.price_increment
            else:
                return possible_buyers[0], lowest_seller, item, \
                    possible_buyers[0].get_stock_quantity(item), current_price

        return

    def solve(self, **kwargs):
        return self.bid(**kwargs)
