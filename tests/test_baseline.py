import sys
import os

# Project root ko Python path me add karo
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from agents.baseline import (
    FixedPricingAgent,
    DiscountPricingAgent,
    RandomPricingAgent
)

state = (100, 30)

fixed = FixedPricingAgent()
discount = DiscountPricingAgent()
random_agent = RandomPricingAgent()

print("Fixed Price Action:", fixed.select_action(state))
print("Discount Action:", discount.select_action(state))
print("Random Action:", random_agent.select_action(state))