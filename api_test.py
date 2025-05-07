# test_api.py
import requests
import json

def test_predict_endpoint():
    url = "http://localhost:5000/predict"
    
    # Test cases
    test_cases = [
        {
            "text": "Government announces new anti-corruption measures",
            "expected_topic": "Corruption"
        },
        {
            "text": "Local sports team wins national championship",
            "expected_topic": "Sports"
        },
        {
            "text": "New environmental protection law passed",
            "expected_topic": "Environment"
        },
        {
            "text": "Women's rights activists protest for equal pay",
            "expected_topic": "Women Rights"
        },
        {
            "text": "Election commission announces dates for upcoming vote",
            "expected_topic": "Election"
        }
    ]
    
    print("Testing predict endpoint with individual requests...")
    
    for i, test_case in enumerate(test_cases):
        try:
            response = requests.post(url, json={"text": test_case["text"]})
            response_data = response.json()
            
            print(f"\nTest Case {i+1}:")
            print(f"Input: {test_case['text']}")
            print(f"Expected Topic: {test_case['expected_topic']}")
            print(f"Predicted Topic: {response_data['predicted_topic']}")
            if 'confidence' in response_data:
                print(f"Confidence: {response_data['confidence']:.4f}")
            
            if response_data['predicted_topic'] == test_case['expected_topic']:
                print("✓ Correct prediction")
            else:
                print("✗ Incorrect prediction")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*50)
    
    # Test batch endpoint
    batch_url = "http://localhost:5000/batch_predict"
    batch_texts = [test_case["text"] for test_case in test_cases]
    
    try:
        print("\nTesting batch_predict endpoint...")
        batch_response = requests.post(batch_url, json={"texts": batch_texts})
        batch_data = batch_response.json()
        
        print("Batch prediction results:")
        for i, prediction in enumerate(batch_data['predictions']):
            print(f"{i+1}. {prediction['original_text']} -> {prediction['predicted_topic']}")
    except Exception as e:
        print(f"Batch prediction error: {e}")

if __name__ == "__main__":
    print("API Test Script")
    print("Make sure the Flask API is running before executing this script")
    print("="*50)
    
    start = input("Press Enter to start testing the API...")
    test_predict_endpoint()