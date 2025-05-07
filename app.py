
from flask import Flask, request, jsonify
import pickle
import re

app = Flask(__name__)


with open('improved_topic_classifier.pkl', 'rb') as f:
    model = pickle.load(f)


def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

@app.route('/')
def home():
    return """
    <html>
        <head>
            <title>Topic Classification API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
                .endpoint { background-color: #e9f7ef; padding: 15px; border-radius: 5px; margin-top: 20px; }
                .example { background-color: #eaf2f8; padding: 15px; border-radius: 5px; margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>Topic Classification API</h1>
            <p>This API classifies text into predefined topics based on a machine learning model.</p>
            
            <div class="endpoint">
                <h2>Predict Endpoint</h2>
                <p>Send a POST request to <code>/predict</code> with JSON data containing the text to classify.</p>
                <pre>
{
    "text": "Your text to classify"
}
                </pre>
            </div>
            
            <div class="example">
                <h2>Example Response</h2>
                <pre>
{
    "original_text": "Government announces new anti-corruption measures",
    "cleaned_text": "government announces new anti corruption measures",
    "predicted_topic": "Corruption",
    "confidence": 0.85
}
                </pre>
            </div>
            
            <div class="example">
                <h2>Example using curl</h2>
                <pre>
curl -X POST http://localhost:5000/predict \\
     -H "Content-Type: application/json" \\
     -d '{"text": "Government announces new anti-corruption measures"}'
                </pre>
            </div>
        </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict_topic():

    data = request.json
    
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    input_text = data['text']
    

    cleaned_text = clean_text(input_text)
    

    predicted_topic = model.predict([cleaned_text])[0]
    
 
    try:
        probabilities = model.predict_proba([cleaned_text])[0]

        confidence = float(max(probabilities))
    except:

        confidence = None
    

    result = {
        'original_text': input_text,
        'cleaned_text': cleaned_text,
        'predicted_topic': predicted_topic
    }
    
    if confidence is not None:
        result['confidence'] = confidence
    
    return jsonify(result)


@app.route('/batch_predict', methods=['POST'])
def batch_predict():

    data = request.json
    
    if 'texts' not in data or not isinstance(data['texts'], list):
        return jsonify({'error': 'Invalid input. Provide a list of texts under "texts" key'}), 400
    
    input_texts = data['texts']
    results = []
    
    for text in input_texts:

        cleaned_text = clean_text(text)
        
        predicted_topic = model.predict([cleaned_text])[0]
        
        results.append({
            'original_text': text,
            'predicted_topic': predicted_topic
        })
    
    return jsonify({'predictions': results})

if __name__ == '__main__':
    print("Starting Topic Classification API...")
    print("API will be available at http://localhost:5000")
    print("To test the API, send a POST request to /predict endpoint")
    app.run(debug=True)