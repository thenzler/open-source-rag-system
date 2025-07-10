#!/usr/bin/env python3
"""
Simple HTTP server to test widget without CORS issues
"""
import http.server
import socketserver
import os
from pathlib import Path

def start_widget_test_server():
    """Start a simple HTTP server for testing the widget"""
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    PORT = 3000
    
    class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Add CORS headers
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
    
    print(f"🚀 Starting widget test server...")
    print(f"📍 Directory: {os.getcwd()}")
    print(f"🌐 Server: http://localhost:{PORT}")
    print(f"🎯 Widget test: http://localhost:{PORT}/widget/")
    print(f"🎯 Example site: http://localhost:{PORT}/example-website/")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")

if __name__ == "__main__":
    start_widget_test_server()