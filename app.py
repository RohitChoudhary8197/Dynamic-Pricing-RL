from flask import Flask, render_template, request

from agents.baseline import DiscountPricingAgent
from env.pricing_env import DynamicPricingEnv
from utils.config import PRICE_LEVELS

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    inventory = int(request.form["inventory"])
    days_left = int(request.form["days_left"])

    env = DynamicPricingEnv()
    agent = DiscountPricingAgent()

    state = [inventory, days_left]
    action = agent.select_action(state)

    price = PRICE_LEVELS[action]
    demand = max(0, min(inventory, (price // 1000) + days_left))

    if price >= 4500:
        demand = max(0, demand - 2)
    if days_left <= 7:
        demand = min(inventory, demand + 3)
    if inventory >= 80:
        demand = min(inventory, demand + 2)

    tickets_sold = min(demand, inventory)
    revenue = tickets_sold * price
    projected_inventory = inventory - tickets_sold

    if projected_inventory < 0:
        projected_inventory = 0

    score = round((revenue / max(1, inventory * price)) * 100, 1)

    return render_template(
        "result.html",
        inventory=inventory,
        days_left=days_left,
        price=f"₹{price:,}",
        revenue=f"₹{revenue:,}",
        demand=tickets_sold,
        projected_inventory=projected_inventory,
        score=score,
        strategy="Adaptive Discount Strategy"
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)