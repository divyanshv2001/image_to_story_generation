from google.cloud import vision
from google.oauth2 import service_account
import openai
openai.api_key = "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" 
def get_image_labels(image_path):
    key_path = "vision-442506-3999f861d947.json"
    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = vision.ImageAnnotatorClient(credentials=credentials)
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    label_response = client.label_detection(image=image)
    labels = label_response.label_annotations
    return [label.description for label in labels]
def generate_story_with_openai(label):
    prompt = f"Write a creative and engaging story based on the following description: {label}. Also check the context with hollywood movies\n\n"    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or you can use "gpt-4" if preferred
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=1,
        top_p=1,
    )
    return response['choices'][0]['message']['content'].strip()
if __name__ == "__main__":
    image_path = "img0.webp"
    labels = get_image_labels(image_path)
    print("\n--- Detected Labels ---")
    for label in labels:
        print(f"- {label}")
    print(generate_story_with_openai(labels))