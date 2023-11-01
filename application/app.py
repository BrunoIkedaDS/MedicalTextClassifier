# This is a simple app to be served as a deployment target for a text classification model builded previously.
# We will be using Flask as the framework due to it's simplicity and ease of use.
# It is going to be a simple web form containing a text input and a submit button, then the model will be used to predict the category of the disease.

# Importing the required libraries.
from flask import Flask, render_template, request
from joblib import load
from utils import preprocess

# Loading the model.
model = load("models/modelo_knn.joblib")

# Creating the app.
app = Flask(__name__)

# Defining the route.
@app.route("/")
def index():
	return render_template("index.html")

# Defining the prediction page route.
@app.route("/predict", methods=["POST"])
def predict():
	text = request.form["text"]

	if text == "":
		return render_template("index.html", string="Please enter a text.")

	text = preprocess(text)
	prediction = model.predict(text)[0]

	# We will return the prediction probability as well.

	prediction_proba = model.predict_proba(text)[0][prediction-1]
	prediction_proba = round(prediction_proba, 2)

	string = "The prediction is: " + str(prediction) + " with a probability of " + str(prediction_proba*100) + "%"

	# Now, our app will throw a prediction on the bottom of the text area.
	return render_template("index.html", string=string)

# Redirecting to the index page.
@app.route("/predict", methods=["GET"])
def get():
	return render_template("index.html")

# Running the app.
if __name__ == "__main__":
	app.run(debug=True)
