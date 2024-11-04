import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from googletrans import Translator
import requests
import json

# Download necessary NLTK data files
nltk.download('punkt')

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

# Preprocess text using NLTK
def preprocess_text(text):
    words = word_tokenize(text)
    # Add your preprocessing steps here
    return ' '.join(words)

# Translate text to English
def translate_to_english(text):
    translator = Translator()
    try:
        result = translator.translate(text, src='bn', dest='en')
        return result.text
    except Exception as e:
        print(f"Error translating to English: {e}")
        return text  # Return the original text if translation fails

# Translate text to Bangla
def translate_to_bangla(text):
    translator = Translator()
    try:
        result = translator.translate(text, src='en', dest='bn')
        return result.text
    except Exception as e:
        print(f"Error translating to Bangla: {e}")
        return text  # Return the original text if translation fails

# Generate response using Mistral Embed and Mistral Large
def generate_response(text, api_key):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        'model': 'ministral-8b-2410',  # Use Mistral Large model
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Please give me the answer from given context ' + text},
        ],
    }
    response = requests.post('https://api.mistral.ai/v1/chat/completions', headers=headers, data=json.dumps(data))

    # Print the response for debugging
    print("API Response:", response.json())

    try:
        return response.json()['choices'][0]['message']['content']
    except KeyError as e:
        print(f"KeyError: {e}")
        return "Error: Unable to generate a response."

# Main function
def main():
    pdf_path = 'chapter-1.pdf'
    api_key = ''

    text = extract_text_from_pdf(pdf_path)
    text = preprocess_text(text)

    # Translate text to English
    english_text = translate_to_english(text)

    # Simulate user query (you can replace this with actual user input)
    # user_query = translate_to_english("সল্প মেয়াদি পদ্ধতি কি এবং এর ধাপ গুলা কি কি")  # Translate this to English if needed
    # user_query = "what is short-term system?"
    user_query = "দীর্ঘ মেয়াদি পদ্ধতি?"

    # Combine the extracted text with the user query
    combined_text = text + " " + user_query

    response = generate_response(combined_text, api_key)

    # Translate the response back to Bangla
    bangla_response = translate_to_bangla(response)

    print(bangla_response)

if __name__ == '__main__':
    main()
