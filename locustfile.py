from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)  # Simulate a user waiting between 1 and 5 seconds between tasks

    @task
    def predict(self):
        data = {'text': "Abbreviations is GEMS (Global Enteric Multi center Study)"}  # Example feature set
        headers = {'content-type': 'application/json'}
        self.client.post("/predict", json=data, headers=headers)
