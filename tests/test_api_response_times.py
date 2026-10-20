#!/usr/bin/env python3
"""Measure API response times and consistency for Chapter 4 evaluation."""

import requests
import time
import json
import os
import sys
import statistics
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "tests" / "chapter4_results"
RESULTS_DIR.mkdir(exist_ok=True)

BASE_URL = "http://127.0.0.1:5000"


def measure_via_test_client(test_image_path, num_requests=10):
    """Fallback: measure response times via Flask test client when the
    live server (and MongoDB auth) are unavailable.  This still exercises
    the full Flask route, preprocessing, and ML inference pipeline."""
    sys.path.insert(0, str(BASE_DIR))
    from app import create_app
    app = create_app()
    client = app.test_client()

    # Set a fake session so the auth check passes
    with client.session_transaction() as sess:
        sess['user_id'] = 'test'
        sess['user_email'] = 'test@test.com'
        sess['user_role'] = 'user'
        sess['user_name'] = 'Test User'

    response_times = []
    results = []

    for i in range(num_requests):
        with open(test_image_path, 'rb') as f:
            start = time.perf_counter()
            resp = client.post('/api/analyze-plant',
                               data={'image': (f, test_image_path.name, 'image/jpeg')},
                               content_type='multipart/form-data')
            elapsed = (time.perf_counter() - start) * 1000
            response_times.append(elapsed)
            if resp.status_code == 200:
                data = resp.get_json()
                results.append({
                    "disease": data.get("disease"),
                    "confidence": data.get("confidence"),
                    "offline": data.get("offline"),
                })
            else:
                results.append({"error": str(resp.status_code)})
            print(f"  Request {i + 1}: {elapsed:.1f}ms - status {resp.status_code}")

    return response_times, results, "flask test client (no MongoDB auth)"


def main():
    session = requests.Session()

    # Login with demo credentials
    print("Logging in...")
    try:
        login_resp = session.post(
            f"{BASE_URL}/login",
            json={"email": "user@agri.com", "password": "password123"},
            timeout=5,
        )
        login_data = login_resp.json()
        print(f"Login response: {login_data}")
    except requests.exceptions.ConnectionError:
        print("No server running at", BASE_URL)
        print("Falling back to Flask test client for response time measurement...")
        login_data = {}

    # Pick a test image
    test_img_dir = BASE_DIR / "data" / "processed" / "split" / "test" / "Tomato_healthy"
    test_images = [f for f in test_img_dir.iterdir()
                   if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    if not test_images:
        print("No test images found for Tomato_healthy")
        return
    img_path = test_images[0]
    print(f"Testing with: {img_path.name}")

    if login_data.get("success"):
        auth_method = "session cookie after login"
        response_times = []
        results = []

        for i in range(10):
            with open(img_path, 'rb') as f:
                files = {"image": (img_path.name, f, "image/jpeg")}
                start = time.perf_counter()
                resp = session.post(f"{BASE_URL}/api/analyze-plant", files=files, timeout=30)
                elapsed = (time.perf_counter() - start) * 1000
                response_times.append(elapsed)
                if resp.status_code == 200:
                    data = resp.json()
                    results.append({
                        "disease": data.get("disease"),
                        "confidence": data.get("confidence"),
                        "offline": data.get("offline"),
                    })
                else:
                    results.append({"error": resp.text[:100]})
                print(f"  Request {i+1}: {elapsed:.1f}ms - status {resp.status_code}")
    else:
        if not login_data:
            pass  # ConnectionError already printed fallback message
        else:
            print("Login failed (MongoDB not available).")
            print("Falling back to Flask test client for response time measurement...")
        response_times, results, auth_method = measure_via_test_client(img_path)

    print(f"\nResponse Time Statistics (ms):")
    print(f"  Mean:   {statistics.mean(response_times):.1f}")
    print(f"  Median: {statistics.median(response_times):.1f}")
    print(f"  Max:    {max(response_times):.1f}")
    print(f"  Min:    {min(response_times):.1f}")
    print(f"  Stdev:  {statistics.stdev(response_times):.1f}")

    # Consistency check
    diseases = [r["disease"] for r in results if "disease" in r]
    consistent = len(set(diseases)) == 1
    print(f"\nConsistency: {consistent} ({len(set(diseases))} unique predictions across 10 runs)")
    print(f"Predicted disease: {diseases[0] if diseases else 'N/A'}")
    if diseases:
        confs = [r["confidence"] for r in results if "confidence" in r]
        print(f"Confidence range: {min(confs):.1f} - {max(confs):.1f}%")

    api_results = {
        "endpoint": "/api/analyze-plant",
        "auth_method": auth_method,
        "response_times_ms": {
            "mean": round(statistics.mean(response_times), 1),
            "median": round(statistics.median(response_times), 1),
            "max": round(max(response_times), 1),
            "min": round(min(response_times), 1),
            "stdev": round(statistics.stdev(response_times), 1),
        },
        "consistency": {
            "consistent": consistent,
            "unique_predictions": len(set(diseases)),
            "unique_disease": diseases[0] if diseases else None,
        },
        "sample_results": results[:3],
    }

    output_path = RESULTS_DIR / "api_response_times.json"
    with open(output_path, "w") as f:
        json.dump(api_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
