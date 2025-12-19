import streamlit as st
import os
import tempfile
import shutil
import subprocess
import time
import csv
import io
from pathlib import Path
from typing import Optional, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    st.error("openai パッケージがインストールされていません。pip install openai を実行してください。")
    st.stop()

# マスカーの読み込み試行
try:
    from masker import load_tokens, mask_text
    _MASK_TOKENS = load_tokens()
    MASKER_AVAILABLE = True
except ImportError:
    MASKER_AVAILABLE = False

# 定数
DEFAULT_MODEL = "whisper-1"
SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".ogg", ".aac", ".flac", ".mpeg", ".mpga"
}

# --- ヘルパー関数 (web_ui.py から移植・調整) ---

def _get_ffmpeg_path() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return shutil.which("ffmpeg")

def _check_ffmpeg() -> bool:
    return _get_ffmpeg_path() is not None

def _compress_audio(input_path: Path) -> Path:
    """
    ffmpegを使用して音声を圧縮する (16kHz, mono, 32kbps MP3).
    """
    tf = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tf.close()
    output_path = Path(tf.name)

    ffmpeg_exe = _get_ffmpeg_path()
    if not ffmpeg_exe:
        raise FileNotFoundError("ffmpegが見つかりません")

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_path),
        "-vn",          # 映像無効化
        "-ar", "16000", # サンプリングレート
        "-ac", "1",     # モノラル
        "-b:a", "32k",  # ビットレート
        str(output_path)
    ]
    
    # ストリームリット上で実行するため、エラー出力をキャプチャして例外に含めるなどの配慮も可能だが
    # ここではシンプルに実装
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    return output_path

# --- UI ---

st.set_page_config(page_title="Whisper API Transcriber", layout="wide")

st.title("Whisper Simple UI (API版)")
st.markdown("OpenAI Whisper API を使用して、音声ファイルを文字起こしします。")

# サイドバー設定
with st.sidebar:
    st.header("設定")
    
    # API Key
    api_key_env = os.environ.get("OPENAI_API_KEY", "")
    api_key_input = st.text_input("OpenAI API Key", value=api_key_env, type="password")
    
    if not api_key_input:
        st.warning("API Key を入力してください（または環境変数 OPENAI_API_KEY を設定）")
    
    # 言語設定
    language = st.text_input("言語コード (例: ja, en)", value="", help="空欄の場合は自動検出")
    
    # マスク設定
    enable_mask = False
    if MASKER_AVAILABLE:
        enable_mask = st.checkbox("個人情報マスクを有効化", value=False)
    else:
        st.caption("masker.py が見つからないため、マスク機能は無効です。")

# メインエリア
uploaded_files = st.file_uploader(
    "音声ファイルをアップロード (複数可)",
    type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
    accept_multiple_files=True
)

if uploaded_files and api_key_input:
    if st.button("文字起こし開始"):
        client = OpenAI(api_key=api_key_input)
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = len(uploaded_files)
        
        # 一時ディレクトリで作業
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path_dir = Path(temp_dir)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                status_text.text(f"処理中 ({idx+1}/{total_files}): {file_name}")
                
                # ファイルを保存
                file_path = temp_path_dir / file_name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                result_data = {
                    "file": file_name,
                    "success": False,
                    "text": "",
                    "text_masked": "",
                    "error": "",
                    "model": DEFAULT_MODEL,
                    "process_time_sec": 0
                }
                
                start_time = time.time()
                compressed_path = None
                target_path = file_path
                
                try:
                    # サイズチェック & 圧縮
                    file_size = file_path.stat().st_size
                    if file_size > 25 * 1024 * 1024:
                        status_text.text(f"圧縮中 ({idx+1}/{total_files}): {file_name} ({file_size/1024/1024:.1f}MB)")
                        try:
                            compressed_path = _compress_audio(file_path)
                            target_path = compressed_path
                        except Exception as e:
                            # 圧縮失敗時はそのままトライするかエラーにするか。web_ui同様、失敗扱いにする
                             raise ValueError(f"25MB超過ファイルの圧縮に失敗しました: {e}")

                    # APIコール
                    status_text.text(f"API送信中 ({idx+1}/{total_files}): {file_name}")
                    with open(target_path, "rb") as audio_file:
                        params = {"model": DEFAULT_MODEL, "file": audio_file}
                        if language.strip():
                            params["language"] = language.strip()
                        
                        resp = client.audio.transcriptions.create(**params)
                    
                    raw_text = getattr(resp, "text", "")
                    masked_text = raw_text
                    if enable_mask and MASKER_AVAILABLE:
                        masked_text = mask_text(raw_text, _MASK_TOKENS)
                    
                    result_data["success"] = True
                    result_data["text"] = raw_text
                    result_data["text_masked"] = masked_text
                    
                except Exception as e:
                    result_data["error"] = str(e)
                finally:
                    result_data["process_time_sec"] = time.time() - start_time
                    if compressed_path and compressed_path.exists():
                        try:
                            compressed_path.unlink()
                        except:
                            pass
                
                results.append(result_data)
                progress_bar.progress((idx + 1) / total_files)
        
        status_text.text("完了！")
        
        # 結果表示
        st.divider()
        st.subheader("処理結果")
        
        # CSV作成
        csv_buffer = io.StringIO()
        fieldnames = ["file", "success", "text", "text_masked", "error", "model", "process_time_sec"]
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        csv_data = csv_buffer.getvalue().encode("utf-8-sig")
        
        st.download_button(
            label="結果CSVをダウンロード",
            data=csv_data,
            file_name="transcripts.csv",
            mime="text/csv"
        )
        
        for res in results:
            with st.expander(f"{res['file']} ({'成功' if res['success'] else '失敗'})"):
                if res["success"]:
                    if enable_mask:
                        st.caption("マスキング済みテキスト")
                        st.write(res["text_masked"])
                        st.caption("原文")
                        with st.expander("原文を表示"):
                            st.write(res["text"])
                    else:
                        st.write(res["text"])
                else:
                    st.error(f"エラー: {res['error']}")
                    
elif not api_key_input:
    st.info("左のサイドバーでAPI Keyを設定してください。")
