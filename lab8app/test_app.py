import requests

sample_input = {
    "features": [1.4, 0.2]
}

url = 'http://127.0.0.1:8000/predict'
response = requests.post(url, json=sample_input)

print("Prediction result:", response.json())