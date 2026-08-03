from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    inventory = request.form["inventory"]
    days_left = request.form["days_left"]

    return render_template(
        "result.html",
        inventory=inventory,
        days_left=days_left,
        price="₹4500",
        revenue="₹425000"
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