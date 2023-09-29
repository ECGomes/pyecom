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
    def get_sorted_sellers(sellers: list[BaseParticipant],
                           item: BaseItem) -> list[BaseParticipant]:
        """
        Get the lowest price of an item
        :param sellers: Seller list
        :param item: Item to be sold
        :return: BaseParticipant or None
        """

        item_sellers = [seller for seller in sellers if seller.get_stock_quantity(seller.sell_stock,
                                                                                  item) > 0]

        # We don't need to sort if there is only one seller
        if len(item_sellers) == 1:
            return item_sellers

        # Sort the sellers by price
        item_sellers.sort(key=lambda x: x.get_stock_quantity(x.sell_stock, item))

        # Return the lowest price
        return item_sellers

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

    def __init__(self, price_increment: float = 0.1, round_limit: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.price_increment = price_increment
        self.round_limit = round_limit

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

        # Get the sellers sorted by price
        possible_sellers = self.get_sorted_sellers(sellers, item)
        lowest_seller = None
        current_price = None
        current_round = 0

        # Check the price of the lowest seller
        if len(possible_sellers) > 0:
            lowest_seller = possible_sellers[0]
            current_price = lowest_seller.get_stock_price(lowest_seller.sell_stock, item)
        else:
            # If there are no sellers, set the lowest price to the baseline price
            # Equivalent to buying from the grid
            current_price = self.baseline_price

        # Get the buyers that want to get the item
        possible_buyers = [buyer for buyer in buyers if buyer.get_stock_quantity(buyer.buy_stock, item) > 0]

        # Iterate while there are buyers
        while len(possible_buyers) > 0 and len(possible_sellers) > 0:

            # If the price is higher than the baseline price, set it to the baseline price
            if current_price >= self.baseline_price:
                current_price = self.baseline_price

            # Check additional sellers if possible
            if len(possible_sellers) > 1:
                next_seller = possible_sellers[1]
                next_price = next_seller.get_stock_price(next_seller.sell_stock, item)
                if next_price < current_price:
                    current_price = next_price
                    lowest_seller = next_seller

            # Get the buyers that can pay the price
            possible_buyers = [buyer for buyer in possible_buyers if buyer.max_bid >= current_price]

            print(f"Current price: {current_price}")
            print(f"Possible buyers: {len(possible_buyers)}")

            # If there is more than, increase the price
            if len(possible_buyers) > 1:
                current_price += self.price_increment
                current_round += 1

                # If the round limit is reached, return the first option
                if current_round >= self.round_limit:
                    return possible_buyers[0], lowest_seller, item, \
                        possible_buyers[0].get_stock_quantity(possible_buyers[0].buy_stock,
                                                              item), current_price
            elif len(possible_buyers) == 1:
                return possible_buyers[0], lowest_seller, item, \
                    possible_buyers[0].get_stock_quantity(possible_buyers[0].buy_stock,
                                                          item), current_price

        print("No buyers or sellers found")

        return None, None, None, 0.0, 0.0

    def solve(self, buyers: list[BaseParticipant],
              sellers: list[BaseParticipant],
              item: BaseItem) -> Union[tuple[BaseParticipant, BaseParticipant, BaseItem, int, float], None]:

        return self.bid(buyers, sellers, item)
