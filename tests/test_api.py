"""
API tests for ML Drilling Operations
"""

import pytest
import json
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.api.app import app

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
def sample_formation_request():
    """Sample formation pressure request"""
    return {
        "well_depth": 5000.0,
        "wob": 25.5,
        "rop": 15.2,
        "torque": 120.0,
        "standpipe_pressure": 2000.0,
        "hook_load": 150.0,
        "differential_pressure": 180.0
    }

@pytest.fixture
def sample_kick_request():
    """Sample kick detection request"""
    return {
        "well_depth": 5000.0,
        "wob": 25.5,
        "rop": 15.2,
        "torque": 120.0,
        "standpipe_pressure": 2000.0,
        "hook_load": 150.0,
        "flow_in": 300.0,
        "flow_out": 305.0,
        "active_pit_volume": 100.5,
        "mud_return_flow": 295.0,
        "block_speed": 50.0
    }

class TestHealthEndpoints:
    """Test health and status endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert data["name"] == "Drilling Operations ML API"
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "loaded_models" in data
        assert "system_info" in data
        assert data["status"] == "healthy"
    
    def test_models_endpoint(self, client):
        """Test models listing endpoint"""
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "formation_pressure" in data
        assert "kick_detection" in data
        assert isinstance(data["formation_pressure"], list)
        assert isinstance(data["kick_detection"], list)

class TestFormationPressureAPI:
    """Test formation pressure prediction API"""
    
    def test_formation_pressure_prediction_success(self, client, sample_formation_request):
        """Test successful formation pressure prediction"""
        response = client.post(
            "/predict/formation-pressure",
            json=sample_formation_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = [
            "predicted_pressure",
            "pressure_gradient", 
            "mud_weight_recommendation",
            "pressure_category",
            "recommendations",
            "timestamp"
        ]
        
        for field in required_fields:
            assert field in data
        
        # Check data types and ranges
        assert isinstance(data["predicted_pressure"], (int, float))
        assert data["predicted_pressure"] > 0
        assert data["predicted_pressure"] < 20000  # Reasonable range
        
        assert isinstance(data["pressure_gradient"], (int, float))
        assert data["pressure_gradient"] > 0
        
        assert isinstance(data["mud_weight_recommendation"], (int, float))
        assert data["mud_weight_recommendation"] > 8  # Minimum mud weight
        assert data["mud_weight_recommendation"] < 20  # Maximum reasonable mud weight
        
        assert data["pressure_category"] in ["Normal", "High", "Low"]
        assert isinstance(data["recommendations"], list)
    
    def test_formation_pressure_invalid_data(self, client):
        """Test formation pressure prediction with invalid data"""
        # Test with negative values
        invalid_request = {
            "well_depth": -1000,  # Invalid negative depth
            "wob": 25.5,
            "rop": 15.2,
            "torque": 120.0,
            "standpipe_pressure": 2000.0,
            "hook_load": 150.0
        }
        
        response = client.post(
            "/predict/formation-pressure",
            json=invalid_request
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_formation_pressure_missing_fields(self, client):
        """Test formation pressure prediction with missing required fields"""
        incomplete_request = {
            "well_depth": 5000.0,
            "wob": 25.5
            # Missing other required fields
        }
        
        response = client.post(
            "/predict/formation-pressure",
            json=incomplete_request
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_formation_pressure_extreme_values(self, client):
        """Test formation pressure prediction with extreme values"""
        extreme_request = {
            "well_depth": 50000,  # Very deep well
            "wob": 100,    # Very high WOB
            "rop": 200,    # Very high ROP
            "torque": 500, # Very high torque
            "standpipe_pressure": 10000,  # Very high pressure
            "hook_load": 500,
            "differential_pressure": 1000
        }
        
        response = client.post(
            "/predict/formation-pressure",
            json=extreme_request
        )
        
        # Should handle extreme values gracefully
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data["predicted_pressure"], (int, float))
        else:
            # Or return appropriate validation error
            assert response.status_code == 422

class TestKickDetectionAPI:
    """Test kick detection API"""
    
    def test_kick_detection_success(self, client, sample_kick_request):
        """Test successful kick detection"""
        response = client.post(
            "/predict/kick-detection",
            json=sample_kick_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = [
            "kick_detected",
            "anomaly_score",
            "confidence_level",
            "risk_level",
            "flow_balance",
            "monitoring_recommendations",
            "timestamp"
        ]
        
        for field in required_fields:
            assert field in data
        
        # Check data types and ranges
        assert isinstance(data["kick_detected"], bool)
        
        assert isinstance(data["anomaly_score"], (int, float))
        assert 0 <= data["anomaly_score"] <= 1
        
        assert data["confidence_level"] in ["High", "Medium", "Low"]
        assert data["risk_level"] in ["Critical", "High", "Medium", "Low"]
        
        assert isinstance(data["flow_balance"], (int, float))
        assert isinstance(data["monitoring_recommendations"], list)
        
        # If kick detected, should have emergency actions
        if data["kick_detected"]:
            assert "emergency_actions" in data
            assert data["emergency_actions"] is not None
            assert isinstance(data["emergency_actions"], list)
    
    def test_kick_detection_high_anomaly_scenario(self, client):
        """Test kick detection with high anomaly indicators"""
        high_anomaly_request = {
            "well_depth": 5000.0,
            "wob": 25.5,
            "rop": 15.2,
            "torque": 120.0,
            "standpipe_pressure": 2000.0,
            "hook_load": 150.0,
            "flow_in": 300.0,
            "flow_out": 350.0,  # High flow out (potential kick indicator)
            "active_pit_volume": 120.0,  # High pit volume (kick indicator)
            "mud_return_flow": 320.0,  # High return flow
            "block_speed": 50.0
        }
        
        response = client.post(
            "/predict/kick-detection",
            json=high_anomaly_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should detect higher risk
        assert data["risk_level"] in ["High", "Critical", "Medium"]
        assert data["anomaly_score"] > 0.1  # Should show some anomaly
    
    def test_kick_detection_flow_balance_calculation(self, client):
        """Test flow balance calculation"""
        test_request = {
            "well_depth": 5000.0,
            "wob": 25.5,
            "rop": 15.2,
            "torque": 120.0,
            "standpipe_pressure": 2000.0,
            "hook_load": 150.0,
            "flow_in": 300.0,
            "flow_out": 310.0,
            "active_pit_volume": 100.0
        }
        
        response = client.post(
            "/predict/kick-detection",
            json=test_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Flow balance should be flow_out - flow_in
        expected_balance = test_request["flow_out"] - test_request["flow_in"]
        assert abs(data["flow_balance"] - expected_balance) < 0.01

class TestBatchPrediction:
    """Test batch prediction endpoints"""
    
    def test_batch_formation_pressure(self, client):
        """Test batch formation pressure prediction"""
        batch_request = {
            "model_type": "formation_pressure",
            "data": [
                {
                    "WellDepth": 5000,
                    "WoBit": 25,
                    "RoPen": 15,
                    "BTBR": 120,
                    "WBoPress": 2000,
                    "HLoad": 150,
                    "DPPress": 180
                },
                {
                    "WellDepth": 5100,
                    "WoBit": 26,
                    "RoPen": 14,
                    "BTBR": 125,
                    "WBoPress": 2050,
                    "HLoad": 155,
                    "DPPress": 185
                }
            ]
        }
        
        response = client.post(
            "/predict/batch",
            json=batch_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_requests" in data
        assert "successful_predictions" in data
        assert "results" in data
        assert data["total_requests"] == 2
        assert len(data["results"]) == 2
        
        # Check individual results
        for result in data["results"]:
            if result["status"] == "success":
                assert "prediction" in result
                assert isinstance(result["prediction"], (int, float))
    
    def test_batch_kick_detection(self, client):
        """Test batch kick detection"""
        batch_request = {
            "model_type": "kick_detection",
            "data": [
                {
                    "FIn": 300,
                    "FOut": 302,
                    "ActiveGL": 100,
                    "WBoPress": 2000,
                    "HLoad": 150
                },
                {
                    "FIn": 300,
                    "FOut": 320,  # Higher flow out
                    "ActiveGL": 110,  # Higher pit volume
                    "WBoPress": 2100,
                    "HLoad": 150
                }
            ]
        }
        
        response = client.post(
            "/predict/batch",
            json=batch_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_requests"] == 2
        assert len(data["results"]) == 2
        
        # Check results structure
        for result in data["results"]:
            if result["status"] == "success":
                assert "kick_detected" in result
                assert "anomaly_score" in result
                assert isinstance(result["kick_detected"], bool)
    
    def test_batch_invalid_model_type(self, client):
        """Test batch prediction with invalid model type"""
        invalid_request = {
            "model_type": "invalid_model",
            "data": [{"field1": 1, "field2": 2}]
        }
        
        response = client.post(
            "/predict/batch",
            json=invalid_request
        )
        
        assert response.status_code == 400  # Bad request

class TestAnalyticsEndpoints:
    """Test analytics and monitoring endpoints"""
    
    def test_formation_pressure_analytics(self, client):
        """Test formation pressure analytics endpoint"""
        response = client.get("/analytics/formation-pressure")
        
        assert response.status_code == 200
        data = response.json()
        
        required_fields = [
            "model_name",
            "analysis_period",
            "prediction_summary",
            "pressure_categories",
            "model_performance"
        ]
        
        for field in required_fields:
            assert field in data
        
        # Check analysis period structure
        assert "start" in data["analysis_period"]
        assert "end" in data["analysis_period"]
        assert "days" in data["analysis_period"]
        
        # Check prediction summary
        summary = data["prediction_summary"]
        assert "total_predictions" in summary
        assert "average_pressure" in summary
        assert "pressure_range" in summary
    
    def test_kick_detection_analytics(self, client):
        """Test kick detection analytics endpoint"""
        response = client.get("/analytics/kick-detection")
        
        assert response.status_code == 200
        data = response.json()
        
        required_fields = [
            "model_name",
            "analysis_period", 
            "detection_summary",
            "risk_distribution",
            "model_performance"
        ]
        
        for field in required_fields:
            assert field in data
        
        # Check detection summary
        summary = data["detection_summary"]
        assert "total_analyses" in summary
        assert "kicks_detected" in summary
        assert "false_positive_rate" in summary
        
        # Check risk distribution
        risk_dist = data["risk_distribution"]
        assert "low" in risk_dist
        assert "medium" in risk_dist
        assert "high" in risk_dist
        assert "critical" in risk_dist
    
    def test_analytics_custom_period(self, client):
        """Test analytics with custom time period"""
        response = client.get("/analytics/formation-pressure?days_back=30")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["analysis_period"]["days"] == 30

class TestErrorHandling:
    """Test API error handling"""
    
    def test_invalid_endpoint(self, client):
        """Test request to invalid endpoint"""
        response = client.get("/invalid/endpoint")
        
        assert response.status_code == 404
    
    def test_invalid_http_method(self, client, sample_formation_request):
        """Test invalid HTTP method"""
        response = client.get(  # GET instead of POST
            "/predict/formation-pressure",
            params=sample_formation_request
        )
        
        assert response.status_code == 405  # Method not allowed
    
    def test_malformed_json(self, client):
        """Test malformed JSON request"""
        response = client.post(
            "/predict/formation-pressure",
            data="invalid json content",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_content_type(self, client, sample_formation_request):
        """Test missing content type header"""
        import json as json_lib
        
        response = client.post(
            "/predict/formation-pressure",
            data=json_lib.dumps(sample_formation_request)
            # Missing content-type header
        )
        
        # Should still work or return appropriate error
        assert response.status_code in [200, 422, 415]

class TestModelManagement:
    """Test model management endpoints"""
    
    def test_model_training_endpoint(self, client):
        """Test model training endpoint"""
        response = client.post(
            "/models/train/formation_pressure",
            params={
                "model_name": "test_model",
                "model_type": "pcr"
            }
        )
        
        # Should accept training request
        assert response.status_code in [200, 202]  # OK or Accepted
        
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "status" in data
    
    def test_model_deletion_nonexistent(self, client):
        """Test deleting nonexistent model"""
        response = client.delete("/models/formation_pressure/nonexistent_model")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["error"].lower()

class TestAPIPerformance:
    """Test API performance requirements"""
    
    def test_prediction_response_time(self, client, sample_formation_request):
        """Test prediction response time"""
        import time
        
        start_time = time.time()
        response = client.post(
            "/predict/formation-pressure",
            json=sample_formation_request
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # API should respond quickly (< 2 seconds)
        assert response_time < 2.0
        assert response.status_code == 200
    
    def test_concurrent_requests(self, client, sample_formation_request):
        """Test handling concurrent requests"""
        import threading
        import time
        
        responses = []
        errors = []
        
        def make_request():
            try:
                response = client.post(
                    "/predict/formation-pressure",
                    json=sample_formation_request
                )
                responses.append(response)
            except Exception as e:
                errors.append(e)
        
        # Start multiple concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all requests to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0  # No errors should occur
        assert len(responses) == 5  # All requests should complete
        
        for response in responses:
            assert response.status_code == 200
    
    def test_large_batch_prediction(self, client):
        """Test handling large batch predictions"""
        # Create large batch request
        large_batch = {
            "model_type": "formation_pressure",
            "data": []
        }
        
        # Add 100 prediction requests
        for i in range(100):
            large_batch["data"].append({
                "WellDepth": 5000 + i,
                "WoBit": 25 + (i % 10),
                "RoPen": 15 + (i % 5),
                "BTBR": 120 + (i % 20),
                "WBoPress": 2000 + (i % 100),
                "HLoad": 150 + (i % 15)
            })
        
        import time
        start_time = time.time()
        
        response = client.post("/predict/batch", json=large_batch)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Should handle large batch reasonably quickly
        assert response_time < 10.0  # Less than 10 seconds
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_requests"] == 100
        assert data["successful_predictions"] > 90  # Most should succeed

class TestAPISecurity:
    """Test API security features"""
    
    def test_sql_injection_attempt(self, client):
        """Test resistance to SQL injection attempts"""
        malicious_request = {
            "well_depth": "5000; DROP TABLE models; --",
            "wob": 25.5,
            "rop": 15.2,
            "torque": 120.0,
            "standpipe_pressure": 2000.0,
            "hook_load": 150.0
        }
        
        response = client.post(
            "/predict/formation-pressure",
            json=malicious_request
        )
        
        # Should reject or sanitize malicious input
        assert response.status_code == 422  # Validation error
    
    def test_xss_attempt(self, client):
        """Test resistance to XSS attempts"""
        xss_request = {
            "well_depth": 5000,
            "wob": "<script>alert('xss')</script>",
            "rop": 15.2,
            "torque": 120.0,
            "standpipe_pressure": 2000.0,
            "hook_load": 150.0
        }
        
        response = client.post(
            "/predict/formation-pressure",
            json=xss_request
        )
        
        # Should reject script tags in numeric fields
        assert response.status_code == 422
    
    def test_oversized_request(self, client):
        """Test handling of oversized requests"""
        # Create oversized batch request
        oversized_request = {
            "model_type": "formation_pressure",
            "data": []
        }
        
        # Add too many requests
        for i in range(10000):  # Very large batch
            oversized_request["data"].append({
                "WellDepth": 5000,
                "WoBit": 25,
                "RoPen": 15,
                "BTBR": 120,
                "WBoPress": 2000,
                "HLoad": 150
            })
        
        response = client.post("/predict/batch", json=oversized_request)
        
        # Should handle gracefully (either process or reject)
        assert response.status_code in [200, 413, 422]  # OK, Payload Too Large, or Validation Error

class TestAPIDocumentation:
    """Test API documentation endpoints"""
    
    def test_openapi_spec(self, client):
        """Test OpenAPI specification endpoint"""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
        
        # Check that our main endpoints are documented
        paths = data["paths"]
        assert "/predict/formation-pressure" in paths
        assert "/predict/kick-detection" in paths
        assert "/health" in paths
    
    def test_docs_endpoint(self, client):
        """Test Swagger UI documentation endpoint"""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_redoc_endpoint(self, client):
        """Test ReDoc documentation endpoint"""
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

# Integration tests that require actual models
class TestIntegrationWithModels:
    """Integration tests with actual ML models"""
    
    @pytest.mark.integration
    def test_end_to_end_formation_pressure(self, client):
        """Test complete end-to-end formation pressure prediction"""
        # This test requires that default models are loaded
        
        request_data = {
            "well_depth": 5000.0,
            "wob": 25.0,
            "rop": 15.0,
            "torque": 120.0,
            "standpipe_pressure": 2000.0,
            "hook_load": 150.0,
            "differential_pressure": 180.0
        }
        
        response = client.post(
            "/predict/formation-pressure",
            json=request_data
        )
        
        if response.status_code == 404:
            pytest.skip("No formation pressure model loaded")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify prediction is reasonable for given inputs
        predicted_pressure = data["predicted_pressure"]
        assert 1000 < predicted_pressure < 10000  # Reasonable pressure range
        
        # Verify gradient calculation
        pressure_gradient = data["pressure_gradient"]
        expected_gradient = predicted_pressure / request_data["well_depth"]
        assert abs(pressure_gradient - expected_gradient) < 0.001
    
    @pytest.mark.integration
    def test_end_to_end_kick_detection(self, client):
        """Test complete end-to-end kick detection"""
        # Normal conditions
        normal_request = {
            "well_depth": 5000.0,
            "wob": 25.0,
            "rop": 15.0,
            "torque": 120.0,
            "standpipe_pressure": 2000.0,
            "hook_load": 150.0,
            "flow_in": 300.0,
            "flow_out": 302.0,
            "active_pit_volume": 100.0
        }
        
        response = client.post(
            "/predict/kick-detection",
            json=normal_request
        )
        
        if response.status_code == 404:
            pytest.skip("No kick detection model loaded")
        
        assert response.status_code == 200
        normal_data = response.json()
        
        # Should likely detect normal conditions
        assert normal_data["risk_level"] in ["Low", "Medium"]
        
        # Anomalous conditions
        anomalous_request = normal_request.copy()
        anomalous_request.update({
            "flow_out": 350.0,  # Much higher flow out
            "active_pit_volume": 130.0  # Higher pit volume
        })
        
        response = client.post(
            "/predict/kick-detection",
            json=anomalous_request
        )
        
        assert response.status_code == 200
        anomalous_data = response.json()
        
        # Should detect higher risk/anomaly
        assert anomalous_data["anomaly_score"] >= normal_data["anomaly_score"]
        assert anomalous_data["flow_balance"] > normal_data["flow_balance"]

# Pytest configuration
@pytest.fixture(scope="session", autouse=True)
def setup_test_api():
    """Setup test environment for API tests"""
    # This runs once before all tests
    import asyncio
    
    # Allow models to load (if they exist)
    asyncio.run(asyncio.sleep(2))
    
    yield
    
    # Cleanup after tests
    pass

# Custom markers for different test categories
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require models)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )

if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short", "-m", "not integration"])