from fastapi.testclient import TestClient
from main import app

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong"}


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 3,
        "sepal_width": 5,
        "petal_length": 3.2,
        "petal_width": 4.4
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() ["flower_class"]=="Iris Virginica"
        assert "datetime" in response.json()

 
# test to check if Iris Virginica is classified correctly
def test_new_testcase1():
    # defining a sample payload for the testcase
     payload = {
        "sepal_length": 5.84,
        "sepal_width": 3.05,
        "petal_length": 3.76,
        "petal_width": 1.20
     }
     with TestClient(app) as client:
          response = client.post("/predict_flower", json=payload)
         # asserting the correct response is received
          assert response.status_code == 200
          assert response.json() ["flower_class"]=="Iris Versicolour"
          assert "datetime" in response.json()


# test to check if Iris Virginica's petal dimensions were classified correctly       
def test_new_testcase2():
    # defining a sample payload for the testcase
     payload = {
        "sepal_length": 5.0,
        "sepal_width": 3.0,
        "petal_length": 1.6,
        "petal_width": 0.2
     }
     with TestClient(app) as client:
          response = client.post("/predict_flower", json=payload)
         # asserting the correct response is received
          assert response.status_code == 200
          assert response.json() ["flower_class"]=="Iris Setosa"
          assert "datetime" in response.json()
