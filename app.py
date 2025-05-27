# app.py
from flask import Flask, request, jsonify
from aniting_ai import get_pet_recommendations

app = Flask(__name__)

@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    input_text = data.get("text")
    
    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    result = get_pet_recommendations(input_text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
