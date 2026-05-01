import http.server
import json
import socketserver
import subprocess
from pathlib import Path

import yaml

PORT = 8000
DIRECTORY = Path(__file__).parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == '/render':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                graph_data = json.loads(post_data)

                # Convert to YAML patch format
                patch = {
                    "name": "Node Editor Patch",
                    "description": "Generated from visual node editor",
                    "sample_rate": 48000,
                    "modules": {},
                    "connections": [],
                    "sequence": [
                        {
                            "time": 0.0,
                            "action": "trigger",
                            "target": "env1" # This might be brittle, need a better way later
                        },
                        {
                            "time": 1.5,
                            "action": "release",
                            "target": "env1"
                        }
                    ]
                }

                # Setup default modules for any node, using their "moduleType" property
                for node_id, node_info in graph_data.get('nodes', {}).items():
                    mod_type = node_info.get('type', 'oscillator')
                    patch['modules'][node_id] = {
                        "type": mod_type,
                        "parameters": node_info.get('parameters', {})
                    }

                    # If it's an envelope, ensure sequence works (hack for now: use first env)
                    if mod_type == 'envelope_adsr':
                        patch['sequence'][0]['target'] = node_id
                        patch['sequence'][1]['target'] = node_id

                # Add connections
                for conn in graph_data.get('connections', []):
                    # We assume out is port 0, in is port 0 for basic,
                    # but should probably handle port numbers. For now just 0.
                    # e.g., from: "osc1.0" to "vca1.0"
                    from_node = conn['from']
                    to_node = conn['to']
                    from_port = conn.get('fromPort', '0')
                    to_port = conn.get('toPort', '0')
                    patch['connections'].append({
                        "from": f"{from_node}.{from_port}",
                        "to": f"{to_node}.{to_port}"
                    })

                # Write patch to temp file
                temp_patch_file = DIRECTORY / "temp_patch.yaml"
                with open(temp_patch_file, 'w') as f:
                    yaml.dump(patch, f, sort_keys=False)

                # Render using uv run
                out_wav = DIRECTORY / "output.wav"

                # Must run from project root to use uv properly
                project_root = DIRECTORY.parent.parent
                result = subprocess.run(
                    ["uv", "run", "python", "render_patch.py", str(temp_patch_file), "--output", str(out_wav)],
                    cwd=project_root,
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0 and out_wav.exists():
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success", "audio_url": "/output.wav"}).encode())
                else:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "error",
                        "message": "Render failed",
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }).encode())

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode())
        else:
            super().do_POST()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving node editor at http://localhost:{PORT}")
        httpd.serve_forever()
