from locust import HttpUser, task, between
import random


class RAGSystemUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts."""
        pass
    
    @task(3)
    def health_check(self):
        """Test health endpoint."""
        self.client.get("/health")
    
    @task(2)
    def api_health_check(self):
        """Test API health endpoint."""
        self.client.get("/api/v1/health")
    
    @task(1)
    def config_endpoint(self):
        """Test config endpoint."""
        with self.client.get("/api/v1/config", catch_response=True) as response:
            if response.status_code == 401:
                # Expected if authentication is required
                response.success()
    
    @task(1)
    def documents_endpoint(self):
        """Test documents endpoint."""
        with self.client.get("/api/v1/documents", catch_response=True) as response:
            if response.status_code == 401:
                # Expected if authentication is required
                response.success()
