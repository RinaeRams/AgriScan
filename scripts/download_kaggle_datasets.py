#!/usr/bin/env python3
"""
AgriScan - Crop Disease Detection System
Main application runner
"""

from app import create_app

app = create_app()

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print("Starting AgriScan Crop Disease Detection System...")
    print(f"Server will be available at: http://localhost:{port}")
    print("Upload plant images to detect diseases using AI")
    app.run(debug=False, host='0.0.0.0', port=port)
