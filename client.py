import requests

url = "http://127.0.0.1:5000/predict"
image_path = "../bear_1_1.jpg"

with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

result = response.json()
print("Prediction:", result['predicted_class'])
#print("Status code:", response.status_code)
#print("Raw response text:", response.text)
