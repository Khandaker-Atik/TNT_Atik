Machine learning-based topic classification system that categorizes text content into predefined topics. The system includes model building, evaluation, improvement, and a Flask API for real-time predictions.

## Project Overview

The system classifies text content into one of the following topics:
- Politics
- Law and Order
- Governance & Policy Reform
- Sports
- Culture & Lifestyle
- Corruption
- International affairs
- Election
- Environment
- Natural Disaster
- Terrorism
- Mob Justice
- Women Rights
- Islamic Fundamentalism
- National Defence
- Diplomacy
- Education
- Religion
- Trade & Commodity Price

## Project Structure

```
.
├── tnt_atik.py or TNT_Atik.ipynb (Google Colab file)  # Script for building and improving the model
├── app.py              # Flask API for serving the model
├── api_test.py         # Script to test the API
├── data_and_topics.xlsx  # Provided Contexts and Topics
├── improved_topic_classifier.pkl  # Saved improved model
├── base_topic_classifier.pkl      # Saved baseline model
├── tfidf_vectorizer.pkl           # Saved TF-IDF vectorizer
├── topic_distribution.png         # Visualization of topic distribution
└── model_improvement_comparison.png  # Visualization of model improvements
```

## Prerequisites

- Python 3.8+
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - flask
  - joblib
  - matplotlib
  - seaborn
  - requests (for testing)

## Installation

1. Clone the repository or download the project files
2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn flask joblib matplotlib seaborn requests

   or pip install -r requirements.txt

   ```
3. dataset file (`data_and_topics.xlsx`) in the project directory 


## Usage

### 1. Building the Model

Run the model building script to process the dataset, train the models, and perform evaluations:

```bash
python model_building.py 
Saved this .py type from google colab directly
(I did the model building in Google Colab thats why given TNT_Atik.ipynb where model building is done and saved.)
Original file is located at
    https://colab.research.google.com/drive/1U86J516040_yLO4ekkZmJt5qg6X1BgZ5
```

This will:
- Clean and preprocess the text data
- Extract topic labels from the training data
- Build a baseline Naive Bayes model
- Build an improved Random Forest model with feature engineering
- Evaluate both models on training, test, and synthetic datasets
- Save the models to disk
- Generate performance comparison visualizations

### 2. Running the API

Start the Flask API server:

```bash
python app.py
```

The API will be available at `http://localhost:5000` with the following endpoints:
- `/` - Home page with API documentation
- `/predict` - POST endpoint for single text classification
- `/batch_predict` - POST endpoint for batch text classification

### 3. Testing the API

Run the test script to test the API with sample texts:

```bash
python api_test.py
```



## API Usage Examples
You must run app.py first in a terminal first then create another terminal and run python api_test.py for the TESTING.

### Single Prediction

```python
import requests

url = "http://localhost:5000/predict"
data = {"text": "Government announces new anti-corruption measures"}

response = requests.post(url, json=data)
print(response.json())
```

Expected response:
```json
{
  "original_text": "Government announces new anti-corruption measures",
  "cleaned_text": "government announces new anti corruption measures",
  "predicted_topic": "Corruption",
  "confidence": 0.85
}
```

### Batch Prediction

```python
import requests

url = "http://localhost:5000/batch_predict"
data = {"texts": [
  "Government announces new anti-corruption measures",
  "Local sports team wins national championship"
]}

response = requests.post(url, json=data)
print(response.json())
```

Expected response:
```json
{
  "predictions": [
    {
      "original_text": "Government announces new anti-corruption measures",
      "predicted_topic": "Corruption"
    },
    {
      "original_text": "Local sports team wins national championship",
      "predicted_topic": "Sports"
    }
  ]
}
```

## Model Improvement Techniques

The improved model achieves better performance through:

1. **Feature Engineering**:
   - Expanded n-gram range from (1,2) to (1,3)
   - Feature selection using chi-squared test
   - Adjusted min_df and max_df parameters

2. **Algorithm Change**:
   - Switched from Naive Bayes to Random Forest

3. **Text Preprocessing Enhancements**:
   - Better handling of hashtags and special characters
   - More sophisticated cleaning logic for multilingual text

4. **Model Pipeline**:
   - Integrated pipeline for feature extraction, selection, and classification

## Performance Metrics

The improved model shows significant gains in precision, recall, and F1-score compared to the baseline model:

- **F1 Score**: Improved by at least 20%
- **Precision**: Improved by at least 20%
- **Recall**: Improved by at least 20%

(Actual improvement percentages will be displayed when running the model_building.py script)

## Future Improvements

- Implement transformer-based models like BERT or mBERT for better handling of multilingual content
- Add language detection to better process mixed-language content
- Expand the synthetic test set for more robust evaluation
- Implement active learning for continuous model improvement
- Add authentication to the API for production use