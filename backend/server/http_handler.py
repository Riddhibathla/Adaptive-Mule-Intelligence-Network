from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler
from urllib.parse import unquote, urlparse

from config import FRONTEND_ROOT
from routes.router import NOT_FOUND, route_get, route_post


class Handler(BaseHTTPRequestHandler):
    server_version = "AMIN/2.0"

    def do_GET(self) -> None:
        payload, status = route_get(self.path)
        if payload is NOT_FOUND:
            self.serve_static(urlparse(self.path).path)
            return
        self.json_response(payload, status)

    def do_POST(self) -> None:
        payload, status = route_post(self.path, self.read_json())
        self.json_response(payload, status)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_common_headers("application/json")
        self.end_headers()

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def json_response(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_common_headers("application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_common_headers(self, content_type: str) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")

    def serve_static(self, path: str) -> None:
        if path == "/":
            path = "/index.html"
        file_path = (FRONTEND_ROOT / unquote(path).lstrip("/")).resolve()
        if not str(file_path).startswith(str(FRONTEND_ROOT)) or not file_path.exists() or not file_path.is_file():
            self.json_response({"error": "not found"}, 404)
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_common_headers(content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return
