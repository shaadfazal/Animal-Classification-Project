from flask import Flask, request, jsonify
import mlflow.pyfunc
from PIL import Image
import torchvision.transforms as transforms
import torch
import json
import io
import numpy as np

app = Flask(__name__)

# Load MLflow model
model = mlflow.pyfunc.load_model("../mlruns/0/9280776d7e4e4133a00aafc59cbfcea6/artifacts/model")

# Load class labels
with open("classes.json", "r") as f:
    class_names = json.load(f)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # shape: (1, 3, 224, 224)
    input_data = tensor.detach().cpu().numpy()  # Convert to numpy array for MLflow model

    prediction = model.predict(input_data)  # Should now work
    predicted_index = int(torch.tensor(prediction).argmax())
    predicted_class = class_names[predicted_index]

    return jsonify({
        "predicted_class": predicted_class,
        "class_index": predicted_index
    })

if __name__ == "__main__":
    app.run(debug=True)
