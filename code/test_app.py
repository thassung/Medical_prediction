import pytest
from flask import Flask
from app import predict_insurance

@pytest.fixture
def app():
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['DEBUG'] = False  # Set to True if you want detailed error messages
    app.add_url_rule('/predict', 'predict_insurance', predict_insurance, methods=['POST'])
    return app

@pytest.fixture
def client(app):
    return app.test_client()

def test_predict_insurance(client):
    # Prepare test data
    test_data = {
        'exercise': 'Daily',
        'junkFood': 'Frequently',
        'smoking': 'No',
        'alcohol': 'Rarely',
        'sedentary': 'Around 6 hrs',
        'stress': 'Medium',
        'drug': 'No',
        'age': 30,
        'sex': 'Male',
        'weight': 70,
        'height': 170,
        'children': 2,
        'region': 'northwest',
        'disease': 'AIDS'  # Replace with an actual disease
    }

    # Make a POST request to the Flask app
    response = client.post('/predict', json=test_data, content_type='application/json')

    # Check if the response status code is 200 (OK)
    assert response.status_code == 200

    # Parse the JSON response
    result = response.get_json()

    # Check if the expected keys are present in the result
    assert 'message' in result
    assert 'type' in result
    assert 'text11_header' in result
    assert 'text11' in result

def test_predict_insurance_invalid(client):
    # Test with invalid input (e.g., missing required fields)
    test_data = {
        'exercise': 'Daily',
        'smoking': 'No',
        'alcohol': 'Rarely',
        'sedentary': 'Around 6 hrs',
        'stress': 'Medium',
        'drug': 'No',
        'age': 30,
        'sex': 'Male',
        'weight': 70,
        'height': 170,
        'children': 2,
        'region': 'northwest',
        'disease': 'AIDS'
    }
    response = client.post('/predict', json=test_data, content_type='application/json')
    assert response.status_code == 200
    result = response.get_json()
    assert 'error' in result
    # Add more specific assertions based on your expected result

def test_predict_insurance_invalid_method(client):
    # Test with an invalid request method (GET instead of POST)
    response = client.get('/predict')
    assert response.status_code == 405  # Method Not Allowed
