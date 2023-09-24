# PyECOM's product to be used in a market
# A product should be comprised of a name, and a price

class BaseItem:

    def __init__(self,
                 identifier: str = 'power',
                 quantity: int = 0,
                 price: float = 0.0):

        self.identifier = identifier
        self.quantity = quantity
        self.price = price

    def cost(self):
        return self.quantity * self.price
