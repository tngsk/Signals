# Signals Project - Standalone Templates

このディレクトリには、プロジェクトで使われている各機能・処理ごとの独立したテンプレートコードが含まれています。これらのテンプレートは、教育目的や独自のプロジェクトでカスタマイズして使用できるように、できる限りシンプルに分割されています。

## 含まれるテンプレート

### 1. `rust_signal_store.rs` (Rust)
ブロックベースのオーディオ信号をモジュール間で共有するためのハブとなる「Signal Store」の最小実装です。
オーディオ処理の基本となるバス（Bus）アーキテクチャを理解するためのコードです。

**実行方法**:
```bash
rustc rust_signal_store.rs && ./rust_signal_store
```

### 2. `rust_module_oscillator.rs` (Rust)
モジュールが実装すべき基本的なインターフェース（`Module` トレイト）と、それを実装したシンプルなサイン波オシレーターの例です。DSP（デジタル信号処理）モジュールの基本構造を学べます。

**実行方法**:
```bash
rustc rust_module_oscillator.rs && ./rust_module_oscillator
```

### 3. `rust_wav_renderer.rs` (Rust)
生成されたオーディオデータ（`f64`の配列）をWAVファイルとして書き出す処理のテンプレートです。
実際にWAVファイルを書き出すには、`hound` クレートへの依存を追加してコメントアウトを外す必要があります。

**実行方法**:
```bash
rustc rust_wav_renderer.rs && ./rust_wav_renderer
```

### 4. `python_simple_server.py` (Python)
外部ライブラリ（FlaskやFastAPIなど）を使用せず、Pythonの標準ライブラリだけで構築されたシンプルなHTTPサーバーです。
フロントエンドのUI（Node Editorなど）からJSONデータを受け取り、処理するためのバックエンドとして機能します。

**実行方法**:
```bash
python python_simple_server.py
```
（別ターミナルから `curl -X POST -H "Content-Type: application/json" -d '{"hello":"world"}' http://127.0.0.1:8000/api/process` でテスト可能）

### 5. `patch_definition.yaml` (YAML)
モジュールの構成、配線（Connections）、時間軸のシーケンス（Sequence）を定義するための宣言的データフォーマットの例です。
このフォーマットをパースすることで、複雑なシンセサイザーの構造をコード外から制御できるようになります。

## カスタマイズと独自プロジェクトへの導入

これらのテンプレートは、余分な依存関係を極力排除しているため、コピー＆ペーストで独自のプロジェクトのベースとしてすぐにお使いいただけます。

- **Rustコード**: 各 `.rs` ファイルは単独でコンパイル・実行可能です。本格的に使用する場合は、`Cargo.toml` を作成してプロジェクトとして管理してください。
- **Pythonコード**: 必要な処理を `do_POST` メソッドの「データに対する独自の処理を行う」箇所に書き加えるだけで、軽量なAPIサーバーが完成します。
