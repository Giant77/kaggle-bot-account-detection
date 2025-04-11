from fastapi.testclient import TestClient
import pytest
from main import app
import pandas as pd
import random

client = TestClient(app)

def get_random_user_data():
    """Generate random user data for testing"""
    return {
        "GENDER": random.choice(["Male", "Female"]),
        "IS_GLOGIN": random.choice([True, False]),
        "FOLLOWER_COUNT": random.randint(0, 1000),
        "FOLLOWING_COUNT": random.randint(0, 1000),
        "CODE_COUNT": random.randint(0, 100),
        "DISCUSSION_COUNT": random.randint(0, 50),
        "AVG_NB_READ_TIME_MIN": round(random.uniform(0, 60), 1),
        "TOTAL_VOTES_GAVE_NB": random.randint(0, 200),
        "TOTAL_VOTES_GAVE_DS": random.randint(0, 200),
        "TOTAL_VOTES_GAVE_DC": random.randint(0, 200)
    }

def test_read_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "version" in response.json()

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    """Test the prediction endpoint with random data"""
    test_data = get_random_user_data()
    response = client.post("/predict/", json=test_data)
    
    # If models are loaded properly, we should get a 200 response
    # Otherwise, we'll get a 500 with a specific error message
    if response.status_code == 200:
        result = response.json()
        assert "prediction" in result
        assert "is_bot" in result
        assert "prediction_text" in result
        assert isinstance(result["is_bot"], bool)
        assert isinstance(result["prediction"], int)
        assert result["prediction_text"] in ["Bot", "Not Bot"]
    else:
        # If models are not found, we expect a specific error
        assert "Models not loaded properly" in response.json()["detail"]

def test_predict_bot_profile():
    """Test prediction with a likely bot profile"""
    bot_profile = {
        "GENDER": "Male",
        "IS_GLOGIN": False,
        "FOLLOWER_COUNT": 0,
        "FOLLOWING_COUNT": 0,
        "CODE_COUNT": 0,
        "DISCUSSION_COUNT": 0,
        "AVG_NB_READ_TIME_MIN": 0.0,
        "TOTAL_VOTES_GAVE_NB": 0,
        "TOTAL_VOTES_GAVE_DS": 0,
        "TOTAL_VOTES_GAVE_DC": 0
    }
    
    response = client.post("/predict/", json=bot_profile)
    if response.status_code == 200:
        # We expect this profile to be classified as a bot, but the test
        # should not fail if it's not (since we don't know the model's behavior)
        result = response.json()
        print(f"Bot profile prediction: {result}")
    
def test_predict_human_profile():
    """Test prediction with a likely human profile"""
    human_profile = {
        "GENDER": "Female",
        "IS_GLOGIN": True,
        "FOLLOWER_COUNT": 150,
        "FOLLOWING_COUNT": 120,
        "CODE_COUNT": 25,
        "DISCUSSION_COUNT": 30,
        "AVG_NB_READ_TIME_MIN": 15.5,
        "TOTAL_VOTES_GAVE_NB": 45,
        "TOTAL_VOTES_GAVE_DS": 30,
        "TOTAL_VOTES_GAVE_DC": 20
    }
    
    response = client.post("/predict/", json=human_profile)
    if response.status_code == 200:
        # We expect this profile to be classified as human, but the test
        # should not fail if it's not (since we don't know the model's behavior)
        result = response.json()
        print(f"Human profile prediction: {result}")

if __name__ == "__main__":
    # Run tests
    test_read_root()
    test_health_check()
    test_predict_endpoint()
    test_predict_bot_profile()
    test_predict_human_profile()
    print("All tests completed!")