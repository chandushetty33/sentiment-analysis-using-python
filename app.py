from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("sentiment_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        # Predict sentiment
        sentiment = model.predict([text])[0]

        # Convert numeric output to text
        sentiment_map = {1: "Positive", -1: "Negative", 0: "Neutral"}
        result = {"sentiment": sentiment_map[sentiment]}

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
