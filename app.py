import os
from flask import Flask, request, render_template
from google.cloud import vision
from google.oauth2 import service_account
import openai
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
google_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# Helper function to get image labels from Google Vision API
def get_image_labels(image_path):
    credentials = service_account.Credentials.from_service_account_file(google_credentials_path)
    client = vision.ImageAnnotatorClient(credentials=credentials)
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    label_response = client.label_detection(image=image)
    labels = label_response.label_annotations
    return [label.description for label in labels]

# Helper function to generate a story using OpenAI
def generate_story_with_openai(labels):
    prompt = f"Write a creative and engaging story based on the following descriptions: {', '.join(labels)}. Also, include Hollywood-style context."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=1,
            top_p=1,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating story: {e}"

# Main route for file upload
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part in the request.'
        file = request.files['file']
        if file.filename == '':
            return 'No file selected.'
        if file:
            os.makedirs('uploads', exist_ok=True)
            file_path = os.path.join('uploads', secure_filename(file.filename))
            file.save(file_path)
            try:
                labels = get_image_labels(file_path)
                story = generate_story_with_openai(labels)
                return render_template('result.html', labels=labels, story=story)
            except Exception as e:
                return f"An error occurred: {e}"
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)
