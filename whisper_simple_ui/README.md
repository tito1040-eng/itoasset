# Whisper Simple UI

ローカルでフォルダ内の音声を OpenAI Whisper API（`whisper-1` 固定）で一括文字起こしし、CSVダウンロードできる簡易UIです。

## セットアップ
```bash
cd /Users/taro.ito/Documents/Work
pip install -r whisper_simple_ui/requirements.txt
export OPENAI_API_KEY="sk-..."  # .env に記載してもOK
```

## 起動
```bash
python whisper_simple_ui/web_ui.py
# ブラウザ: http://127.0.0.1:8000
```

## 画面で設定できる項目
- フォルダアップロード（ブラウザでフォルダ選択: `webkitdirectory`）
- 入力フォルダ（サーバーパスを直接入力する場合）
- 言語コード（空なら自動）
- 並列数（workers）
- 出力フォルダ（UIでDownloads/Documents/tmpから選択 or カスタム指定。未指定なら `<input_dir>/transcripts.csv`）
- マスク有効化（チェックした場合のみ。デフォルト無効）

## 備考
- モデルは `whisper-1` に固定されています（`WHISPER_MODEL` 環境変数で上書き可）。
- マスク機能は任意。不要ならチェックを外してください。

