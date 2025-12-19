import streamlit as st
import whisper
import tempfile
import os

st.title("Whisper Simple UI (Streamlit版)")

st.markdown("Whisper を使って音声を文字起こしします。")

# モデル選択
model_name = st.selectbox(
    "モデルを選択してください",
    ["tiny", "base", "small", "medium", "large"]
)

# オプション選択（元コードから移植）
language = st.text_input("言語コード（未入力なら自動判定）", "")
temperature = st.slider("temperature", 0.0, 1.0, 0.0)

uploaded_file = st.file_uploader(
    "音声ファイルをアップロードしてください",
    type=["mp3", "wav", "m4a"]
)

if uploaded_file is not None:
    st.write("アップロード完了：解析中…")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # モデル読み込み
    model = whisper.load_model(model_name)

    # パラメータ設定
    options = {}
    if language.strip() != "":
        options["language"] = language
    options["temperature"] = temperature

    # 文字起こし
    result = model.transcribe(temp_path, **options)

    st.subheader("Transcription")
    st.write(result["text"])

    # 不要ファイル削除
    os.remove(temp_path)
