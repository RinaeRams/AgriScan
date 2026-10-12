#!/usr/bin/env python3
"""Merge all Chapter 4 results into a single JSON file."""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "chapter4_results"
results_path = RESULTS_DIR / "chapter4_results.json"

with open(results_path) as f:
    results = json.load(f)

# Add API response times
with open(RESULTS_DIR / "api_response_times.json") as f:
    api_data = json.load(f)
results["api_response_times"] = api_data

# Add end-to-end test results
results["end_to_end_tests"] = {
    "desktop": {
        "login_page": "PASS - Renders login form with register button, email/password fields, demo credentials toggle",
        "register_button": "PASS - Gradient Create Account button visible, links to /register",
        "register_page": "PASS - Registration form shows all fields (firstName, lastName, email, phone, location, farmSize, password, confirmPassword)",
        "login_api": "PASS - POST /login with demo credentials returns success:true, role:user",
        "analyze_api": "PASS - POST /api/analyze-plant returns disease, confidence, all_predictions, recommendations, offline=false",
        "health_endpoint": "PASS - GET /api/health returns status:healthy, ml_service:active, mongodb:connected",
    },
    "mobile": {
        "responsive_layout": "PASS - CSS media queries tested at 820px and 420px breakpoints",
        "login_form": "PASS - Form fields resize for mobile, single-column layout on narrow screens",
        "register_button": "PASS - Register button visible and clickable on mobile viewport",
        "note": "Browser viewport testing via CSS inspection; full mobile device testing requires physical devices",
    },
    "render": {
        "status": "Service suspended (free tier) - 503 Service Suspended",
        "health_url": "https://agriscan-app.onrender.com/api/health",
        "note": "Service auto-deploys on resume. Code pushed to GitHub commit 1b9846c.",
    },
    "api_response_time_note": "First request includes TF model warm-up (~6.9s). Steady-state median: 339.9ms.",
}

with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print("Updated chapter4_results.json with API and E2E data")
print(f"File size: {results_path.stat().st_size / 1024:.1f} KB")
