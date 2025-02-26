from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle 

app = Flask(__name__)
data = pd.read_csv("my_data.csv")  # Replace with your CSV file path
pipe = pickle.load(open("RidgeModel.pkl", "rb"))  # Replace with your model file path

@app.route("/", methods=["GET"])
def index():
    locations= sorted(data['location'].unique())
    return render_template("index.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or request.form  # Handle both JSON and form data
        location = data.get("location")
        bhk = int(data.get("bhk"))
        bath = int(data.get("bathrooms"))
        sqft = float(data.get("sqft"))

        input_df = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = pipe.predict(input_df)[0] * 1e5

        return jsonify({"price": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
