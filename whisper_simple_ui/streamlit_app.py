import streamlit as st
import os
import tempfile
import shutil
import subprocess
import time
import csv
import io
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    _MASK_TOKENS = None

# 定数
DEFAULT_MODEL = "whisper-1"
SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".ogg", ".aac", ".flac", ".mpeg", ".mpga"
}

# --- ヘルパー関数 ---

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
    # 拡張子を.mp3にした一時ファイルを作成
    # delete=False にして、パスを取得した後にクローズし、ffmpegに渡す
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
    
    # ffmpeg実行
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    return output_path

def process_single_file(
    file_path: Path,
    original_name: str,
    api_key: str,
    language: Optional[str],
    enable_mask: bool
) -> Dict[str, Any]:
    """
    1つのファイルを処理するワーカー関数
    """
    client = OpenAI(api_key=api_key)
    
    result_data = {
        "file": original_name,
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
        try:
            file_size = file_path.stat().st_size
        except Exception:
            file_size = 0

        # 25MB (approx 26,214,400 bytes)
        if file_size > 25 * 1024 * 1024:
            # 圧縮試行
            try:
                compressed_path = _compress_audio(file_path)
                target_path = compressed_path
            except Exception as e:
                raise ValueError(f"25MB超過ファイルの圧縮に失敗しました: {e}")

        # APIコール
        with open(target_path, "rb") as audio_file:
            params = {"model": DEFAULT_MODEL, "file": audio_file}
            if language and language.strip():
                params["language"] = language.strip()
            
            resp = client.audio.transcriptions.create(**params)
        
        raw_text = getattr(resp, "text", "")
        masked_text = raw_text
        if enable_mask and MASKER_AVAILABLE and _MASK_TOKENS:
            masked_text = mask_text(raw_text, _MASK_TOKENS)
        
        result_data["success"] = True
        result_data["text"] = raw_text
        result_data["text_masked"] = masked_text
        
    except Exception as e:
        result_data["error"] = str(e)
    finally:
        result_data["process_time_sec"] = time.time() - start_time
        # 圧縮ファイルの削除
        if compressed_path and compressed_path.exists():
            try:
                compressed_path.unlink()
            except:
                pass
                
    return result_data

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
    
    # ワーカー数
    workers = st.number_input("並列処理数 (Workers)", min_value=1, max_value=10, value=4)

    # マスク設定
    enable_mask = False
    if MASKER_AVAILABLE:
        enable_mask = st.checkbox("個人情報マスクを有効化", value=False)
    else:
        st.caption("masker.py が見つからないため、マスク機能は無効です。")
        st.caption("requirements.txtの依存関係を確認してください。")

# メインエリア
uploaded_files = st.file_uploader(
    "音声ファイルをアップロード (複数可)",
    type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
    accept_multiple_files=True
)

if uploaded_files and api_key_input:
    if st.button("文字起こし開始"):
        # プログレスバーなどのUI要素準備
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total_files = len(uploaded_files)
        
        # 一時ディレクトリを作成して処理
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path_dir = Path(temp_dir)
            status_text.text("ファイルを準備中...")
            
            # 1. まず全てのファイルを一時ディレクトリに保存する (並列処理のためにファイル実体が必要)
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = temp_path_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append((file_path, uploaded_file.name))
            
            # 2. 並列処理実行
            completed_count = 0
            status_text.text(f"処理中 (0/{total_files})... 並列数: {workers}")
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # タスクのサブミット
                future_to_file = {
                    executor.submit(
                        process_single_file, 
                        path, 
                        name, 
                        api_key_input, 
                        language, 
                        enable_mask
                    ): name for path, name in file_paths
                }
                
                # 完了したものから順次処理
                for future in as_completed(future_to_file):
                    name = future_to_file[future]
                    try:
                        res = future.result()
                        results.append(res)
                    except Exception as e:
                        # 万が一の予期せぬエラー
                        results.append({
                            "file": name,
                            "success": False,
                            "text": "",
                            "text_masked": "",
                            "error": f"予期せぬエラー: {str(e)}",
                            "model": DEFAULT_MODEL,
                            "process_time_sec": 0
                        })
                    
                    completed_count += 1
                    progress = completed_count / total_files
                    progress_bar.progress(progress)
                    status_text.text(f"処理中 ({completed_count}/{total_files}): 最新の完了 -> {name}")
        
        status_text.text("完了！")
        
        # --- 結果表示 ---
        st.divider()
        st.subheader("処理結果")
        
        # 名前順にソートして表示
        results.sort(key=lambda x: x["file"])
        
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
        
        # 個別結果の表示
        for res in results:
            label = f"{res['file']}"
            if not res['success']:
                label += " ❌ (失敗)"
            else:
                label += " ✅ (成功)"
                
            with st.expander(label):
                st.write(f"処理時間: {res['process_time_sec']:.2f}秒")
                if res["success"]:
                    if enable_mask:
                        st.markdown("**マスキング済みテキスト:**")
                        st.write(res["text_masked"])
                        st.markdown("**原文:**")
                        with st.expander("原文を表示"):
                            st.write(res["text"])
                    else:
                        st.write(res["text"])
                else:
                    st.error(f"エラー内容: {res['error']}")
                    
elif not api_key_input:
    st.info("左のサイドバーでAPI Keyを設定してください。")
