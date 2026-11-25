"""Exercise the ComfyUI HTTP/protocol boundary without model weights."""
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import threading
import unittest
import os

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6G1sAAAAASUVORK5CYII=")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, value):
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/object_info":
            return self.reply({name: {} for name in ("AILab_RMBG3", "UNETLoader", "CLIPLoader", "VAELoader")})
        if self.path == "/history/job1":
            return self.reply({"job1": {"outputs": {"6": {"images": [{"filename": "matte.png"}]}}}})
        if self.path.startswith("/view?"):
            self.send_response(200)
            self.end_headers()
            return self.wfile.write(PNG)
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = self.rfile.read(length)
        if self.path == "/upload/image":
            assert PNG in payload
            return self.reply({"name": "input.png"})
        if self.path == "/prompt":
            graph = json.loads(payload)["prompt"]
            assert graph["2"]["inputs"]["unet_name"] == "rmbg3_int8_convrot.safetensors"
            assert graph["5"]["inputs"]["steps"] == 4
            assert graph["6"]["inputs"]["images"] == ["5", 2]
            return self.reply({"prompt_id": "job1"})
        self.send_error(404)

class ComfyWorkerTest(unittest.TestCase):
    def test_protocol(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env = os.environ.copy()
        env["RMBG_COMFY_URL"] = f"http://127.0.0.1:{server.server_port}"
        worker = subprocess.Popen(["python3", "-u", str(Path(__file__).with_name("comfy_worker.py"))],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        try:
            self.assertTrue(json.loads(worker.stdout.readline())["ready"])
            worker.stdin.write(json.dumps({"image": base64.b64encode(PNG).decode()}) + "\n")
            worker.stdin.flush()
            self.assertEqual(base64.b64decode(json.loads(worker.stdout.readline())["matte"]), PNG)
        finally:
            worker.kill()
            worker.wait()
            worker.stdin.close()
            worker.stdout.close()
            server.shutdown()
            server.server_close()

if __name__ == "__main__":
    unittest.main()
