"""
MindPulse AI - Lightweight Python Application Server.
Serves static web assets and handles real-time psychometric inference API.
Built entirely with standard library and Scikit-learn backend.
"""

import http.server
import json
import os
import sys
from typing import Dict, Any

from src.predictor import MindPulsePredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

# Initialize Scikit-learn predictor instance
print("[INFO] Initializing MindPulse Scikit-learn Predictor Engine...")
predictor = MindPulsePredictor()
print("[INFO] Models loaded successfully!")


class MindPulseHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP Request Handler for static files and REST API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_POST(self):
        if self.path == "/api/predict":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(raw_body)

                responses = payload.get("responses", [])
                if len(responses) != 21:
                    self._send_json({"error": "Expected 21 questionnaire item responses"}, status=400)
                    return

                # Perform prediction
                result = predictor.predict(responses)
                self._send_json(result, status=200)

            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
        else:
            self.send_error(404, "Endpoint not found")

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = http.server.HTTPServer(server_address, MindPulseHandler)
    print("=" * 65)
    print(f"  MINDPULSE AI WEB APPLICATION RUNNING")
    print(f"  Access in your browser at: http://localhost:{port}")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
