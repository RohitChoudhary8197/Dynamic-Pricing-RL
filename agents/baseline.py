"""
Baseline Pricing Strategies

These strategies are used as benchmarks
to compare the Reinforcement Learning agent.
"""

from utils.config import PRICE_LEVELS


class FixedPricingAgent:
    """
    Always selects the same price.
    """

    def __init__(self, fixed_price_index=2):
        """
        fixed_price_index

        0 -> 3000
        1 -> 3500
        2 -> 4000
        3 -> 4500
        4 -> 5000
        """

        self.fixed_price_index = fixed_price_index

    def select_action(self, state):

        return self.fixed_price_index


class DiscountPricingAgent:
    """
    Gradually reduces prices as
    departure day approaches.
    """

    def select_action(self, state):

        inventory, days_left = state

        if days_left > 25:

            return 4

        elif days_left > 20:

            return 3

        elif days_left > 10:

            return 2

        elif days_left > 5:

            return 1

        else:

            return 0


class RandomPricingAgent:
    """
    Chooses a random price level.
    """

    def select_action(self, state):

        import random

        return random.randint(
            0,
            len(PRICE_LEVELS) - 1
        )