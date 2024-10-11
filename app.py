import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import ClientError
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
import heapq
# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Configure AWS credentials and S3 bucket
S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'eu-west-2')

print("S3_BUCKET", S3_BUCKET)
# Initialize S3 client
s3_client = boto3.client('s3', AWS_REGION)

uploaded_files = []

def load_existing_files():
    global uploaded_files
    try:
        response = s3_client.list_objects_v2(Bucket="sagemaker-eu-west-2-307946670775")
        if 'Contents' in response:
            uploaded_files = [obj['Key'] for obj in response['Contents']]
    except ClientError as e:
        print(f"Error loading existing files: {e}")
# Load existing files when the application starts
load_existing_files()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        try:
            s3_client.upload_fileobj(file, S3_BUCKET, filename)
            if filename not in uploaded_files:
                uploaded_files.append(filename)
            return jsonify({'message': 'File uploaded successfully', 'filename': filename, 'files': uploaded_files}), 200
        except ClientError as e:
            return jsonify({'error': str(e)}), 500

@app.route('/get_files', methods=['GET'])
def get_files():
    return jsonify({'files': uploaded_files})

@app.route('/summarize', methods=['POST'])
def summarize_document():
    filename = request.json.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    try:
        # Read the file from S3
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=filename)
        file_content = response['Body'].read().decode('utf-8')

        # Summarize the content
        summary = summarize_text(file_content)

        return jsonify({'summary': summary}), 200
    except ClientError as e:
        return jsonify({'error': str(e)}), 500

def summarize_text(text, num_sentences=5):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Calculate word frequencies
    freq = FreqDist(words)

    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = freq[word]
                else:
                    sentence_scores[sentence] += freq[word]

    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # Join the sentences to create the summary
    summary = ' '.join(summary_sentences)

    return summary



if __name__ == '__main__':
    app.run(debug=True)



# from flask import Flask, render_template, request, jsonify
# from RagUsingAmazonBedrock import chatbot_response
# import random

# app = Flask(__name__)

# # Simulated chatbot responses
# responses = [
#     "That's an interesting question!",
#     "I'm not sure about that. Can you tell me more?",
#     "Let me think about that for a moment...",
#     "That's a great point you've raised.",
#     "I'd need more information to answer that accurately.",
#     "Have you considered looking at it from a different perspective?",
#     "That's a complex topic with many facets to consider.",
#     "I'm glad you asked that. It's an important question.",
# ]

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         prompt = request.form['prompt']
#         # Here you would typically process the prompt and generate a response
#         # For now, we'll just return a random response
#         response = chatbot_response(prompt)
#         #print("response...............", response)
#         #response = random.choice(responses)
#         return jsonify({'response': response})
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)