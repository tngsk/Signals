"""
依存関係なし（標準ライブラリのみ）で動作するシンプルなHTTPサーバー。
UIフロントエンドなどからJSONデータを受け取り、処理するバックエンドのテンプレートです。
"""

import http.server
import json
import socketserver

# サーバーの設定
PORT = 8000
HOST = "127.0.0.1" # セキュリティのためローカルループバックインターフェースに限定

class SimpleJSONHandler(http.server.SimpleHTTPRequestHandler):
    """POSTリクエストでJSONを受け取るハンドラ"""

    def do_POST(self):
        # /api/process エンドポイントの処理
        if self.path == '/api/process':
            # リクエストボディからJSONデータを読み取る
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)

            try:
                # JSONをパース
                data = json.loads(post_data)
                print(f"Received data: {data}")

                # --- ここでデータに対する独自の処理を行う ---
                # 例: データの加工や、別のシステムへの保存など
                response_message = f"Successfully processed {len(data.keys())} items."

                # クライアントへ成功レスポンス(200 OK)を返す
                self.send_response(200)
                self.send_header('Content-type', 'application/json')

                # CORS対応が必要な場合 (異なるポートからのリクエストを許可)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                # レスポンスデータをJSONとして送信
                response_data = {
                    "status": "success",
                    "message": response_message,
                    "original_data": data
                }
                self.wfile.write(json.dumps(response_data).encode('utf-8'))

            except json.JSONDecodeError:
                # JSONのパースエラー時のレスポンス(400 Bad Request)
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {"status": "error", "message": "Invalid JSON data"}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))

            except Exception as e:
                # その他のエラー時のレスポンス(500 Internal Server Error)
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {"status": "error", "message": str(e)}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
        else:
            # 指定されたパス以外は、通常のファイルサーバーとして動作させるかエラーを返す
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found")

    def do_OPTIONS(self):
        """CORSのプリフライトリクエストに対応"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

if __name__ == "__main__":
    # サーバーの起動
    print(f"Starting simple HTTP server at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")

    with socketserver.TCPServer((HOST, PORT), SimpleJSONHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.server_close()
