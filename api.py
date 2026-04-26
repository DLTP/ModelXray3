from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# ---------- LOAD DATA ----------
df = pd.read_csv('calculator_data.csv')

# operation encoding
m = {
    '+': [1,0,0],
    '-': [0,1,0],
    '*': [0,0,1]
}

# ---------- MODEL ----------
model = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# ---------- SAVE / LOAD ----------
MODEL_FILE = "model_weights.pth"

def save_model():
    torch.save(model.state_dict(), MODEL_FILE)

def load_model():
    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))
        print("Model loaded from file")
    else:
        print("No saved model found, using random weights")

load_model()

# ---------- TRAIN FULL ----------
@app.route('/train_full', methods=['POST'])
def train_full():

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(800):
        for _, row in df.iterrows():

            x = torch.tensor([
                row['a'] / 20,
                *m[row['op']],
                row['b'] / 20
            ], dtype=torch.float32)

            y = torch.tensor([row['result'] / 20], dtype=torch.float32)

            pred = model(x)
            loss = (pred - y) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    save_model()

    return jsonify({"status": "trained fully"})


# ---------- TRAIN STEP (FOR TEACHING) ----------
@app.route('/train_step', methods=['POST'])
def train_step():

    row_index = int(request.json.get('row', 0))
    row = df.iloc[row_index]

    x = torch.tensor([
        row['a'] / 20,
        *m[row['op']],
        row['b'] / 20
    ], dtype=torch.float32)

    y = torch.tensor([row['result'] / 20], dtype=torch.float32)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # BEFORE
    before = [p.detach().numpy().tolist() for p in model.parameters()]

    pred = model(x)
    loss = (pred - y) ** 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # AFTER
    after = [p.detach().numpy().tolist() for p in model.parameters()]

    save_model()

    return jsonify({
        "before": before,
        "after": after,
        "input": x.tolist(),
        "prediction": pred.item() * 20,
        "target": y.item() * 20
    })


# ---------- GET WEIGHTS (FOR UI DISPLAY) ----------
@app.route('/get_weights', methods=['GET'])
def get_weights():
    weights = [p.detach().numpy().tolist() for p in model.parameters()]
    return jsonify(weights)


# ---------- PREDICT WITH STEP-BY-STEP ----------
@app.route('/predict', methods=['POST'])
def predict():

    d = request.json

    x = torch.tensor([
        d['a'] / 20,
        *m[d['op']],
        d['b'] / 20
    ], dtype=torch.float32)

    steps = []
    current = x

    # forward pass step-by-step
    for layer in model:

        if isinstance(layer, nn.Linear):
            weights = layer.weight.detach().numpy().tolist()
            bias = layer.bias.detach().numpy().tolist()

            output = layer(current)

            steps.append({
                "type": "linear",
                "input": current.detach().numpy().tolist(),
                "weights": weights,
                "bias": bias,
                "output": output.detach().numpy().tolist()
            })

            current = output

        elif isinstance(layer, nn.ReLU):

            output = layer(current)

            steps.append({
                "type": "activation",
                "input": current.detach().numpy().tolist(),
                "output": output.detach().numpy().tolist()
            })

            current = output

    final = current.item() * 20

    return jsonify({
        "result": round(final, 2),
        "steps": steps
    })


# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)