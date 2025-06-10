from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your model
model = pickle.load(open("best_nids_model.pkl", "rb"))

# Features expected
selected_features = [
    "Flow Duration", "Protocol", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Fwd Pkt Len Max", "Bwd Pkt Len Max", "Flow Byts/s", "Flow Pkts/s",
    "Hour", "Minute", "Second"
]

# Labels
lab = ['Benign', 'FTP-BruteForce', 'SSH-BruteForce']


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Get form values
        values = [float(request.form[feature]) for feature in selected_features]

        # 2. Convert to NumPy array
        input_array = np.array(values).reshape(1, -1)

        # 3. Make prediction
        prediction = model.predict(input_array)[0]

        # 👉👉 4. DEBUG LINE — SEE WHAT THE MODEL PREDICTS
        print("Predicted class index:", prediction)

        # 5. Show result on webpage
        return render_template("index.html", prediction=lab[prediction])

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
