import os
import requests
import json
import base64

# Read the OpenAI API key from the environment variable
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    print("OpenAI API key not found.")
    exit()

# Prompt for DallE
prompt = "A flower in the ocean"

# Define the API endpoint
url = "https://api.openai.com/v1/engines/davinci/codex/completions"

# Define the headers
headers = {
    "Authorization": "Bearer " + openai_key,
    "Content-Type": "application/json"
}

# Define the data payload
data = {
    "prompt": prompt,
    "max_tokens": 32
}

# Send the request to DallE
response = requests.post(url, headers=headers, json=data)
response_data = json.loads(response.content)

# Check if the request was successful
if response.status_code == 200:
    # Get the generated image URL from the response
    image_url = response_data["choices"][0]["text"]
    
    # Download the image
    image_response = requests.get(image_url)
    
    # Save the image to a file
    with open("image.jpg", "wb") as f:
        f.write(image_response.content)
    
    print("Image saved as image.jpg")
else:
    print("Error:", response_data["error"]["message"])