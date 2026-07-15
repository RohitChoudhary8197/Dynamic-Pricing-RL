import random
def calculate_demand(price, days_left):
    """
    Calculate customer demand based on price and remaining days.
    """

    if price == 3000:
        base = 12

    elif price == 3500:
        base = 10

    elif price == 4000:
        base = 8

    elif price == 4500:
        base = 5

    else:
        base = 3

    if days_left <= 5:
        base += 4

    elif days_left <= 10:
        base += 2

    demand = random.randint(max(0, base - 2), base + 2)

    return demand