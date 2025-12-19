#!/usr/bin/env python3
"""
ローカル管理画面: フォルダ内の音声を OpenAI Whisper API で一括文字起こしし、CSVをダウンロード。

起動:
  export OPENAI_API_KEY="sk-..."
  pip install -r whisper_simple_ui/requirements.txt
  python whisper_simple_ui/web_ui.py
  # ブラウザで http://127.0.0.1:8000 を開く
"""

import os
import sys
import time
import uuid
import shutil
import tempfile
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_file

# .env を任意で読み込み（存在しない場合は無視）
try:
    from dotenv import load_dotenv

    load_dotenv()
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
except Exception:
    pass

try:
    from openai import OpenAI
except Exception:
    print(
        "依存関係が不足しています。まず 'pip install -r whisper_simple_ui/requirements.txt' を実行してください。",
        file=sys.stderr,
    )
    sys.exit(1)

# マスカー（デフォルト無効）
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from masker import load_tokens, mask_text  # type: ignore

    _MASK_TOKENS = load_tokens()
except Exception:
    load_tokens = None  # type: ignore
    mask_text = None  # type: ignore
    _MASK_TOKENS = None  # type: ignore

DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "whisper-1")

SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".webm",
    ".ogg",
    ".aac",
    ".flac",
    ".mpeg",
    ".mpga",
}

app = Flask(__name__)
_job_lock = threading.Lock()
_job_state: Dict[str, Any] = {
    "id": None,
    "status": "idle",  # idle | running | done | error
    "progress": {"current": 0, "total": 0},
    "log": [],
    "csv_path": None,
    "params": {},
    "error": "",
    "started_at": None,
    "ended_at": None,
}


def _log(message: str) -> None:
    with _job_lock:
        logs = _job_state.get("log", [])
        logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        # keep last 300 lines
        _job_state["log"] = logs[-300:]


def discover_audio_files(input_dir: str) -> List[Path]:
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(f"入力ディレクトリが見つかりません: {input_dir}")
    files = [p for p in base.rglob("*") if _is_supported_audio_path(p)]
    return sorted(files)


def write_csv(results: List[Dict[str, Any]], csv_path: Path) -> None:
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "success",
        "text",
        "text_masked",
        "error",
        "model",
        "language",
        "process_time_sec",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _normalize_user_path(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    try:
        return str(Path(path_str).expanduser())
    except Exception:
        return path_str


def _is_hidden_path(p: Path) -> bool:
    # macOS の "._xxxx" なども含めて除外
    name = p.name
    return name.startswith(".") or name.startswith("._")


def _is_supported_audio_path(p: Path) -> bool:
    return p.is_file() and (not _is_hidden_path(p)) and (p.suffix.lower() in SUPPORTED_EXTENSIONS)


def _get_ffmpeg_path() -> Optional[str]:
    # 1. Try imageio_ffmpeg (bundled binary)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    # 2. Try system ffmpeg
    return shutil.which("ffmpeg")


def _check_ffmpeg() -> bool:
    return _get_ffmpeg_path() is not None


def _save_uploaded_files(file_storages, base_dir: Path) -> List[Tuple[Path, str, str, int]]:
    """
    Save uploaded files (with webkitdirectory) into base_dir, preserving relative paths.
    """
    saved: List[Tuple[Path, str, str, int]] = []
    base_dir.mkdir(parents=True, exist_ok=True)
    base_resolved = base_dir.resolve()
    for storage in file_storages:
        rel = storage.filename or "uploaded"
        # 安全化（パストラバーサル対策）
        rel_parts = [p for p in Path(rel).parts if p not in ("..", "", "/")]
        if not rel_parts:
            rel_parts = [storage.filename or "unknown_file"]

        target = base_dir.joinpath(*rel_parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved = target.resolve()

        if str(resolved).startswith(str(base_resolved)):
            storage.save(resolved)
            mime = getattr(storage, "mimetype", "") or ""
            orig = storage.filename or ""
            try:
                size = resolved.stat().st_size
            except Exception:
                size = -1
            saved.append((resolved, orig, mime, size))
        else:
            # 危険なパスはスキップ
            continue
    return saved


def _compress_audio(input_path: Path) -> Path:
    """
    ffmpegを使用して音声を圧縮する (16kHz, mono, 32kbps MP3).
    """
    # 拡張子を.mp3にした一時ファイルを作成
    tf = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tf.close()
    output_path = Path(tf.name)

    # ffmpeg -y -i input -vn -ar 16000 -ac 1 -b:a 32k output.mp3
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
    
    # 実行 (エラー時は例外送出)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    return output_path


def _run_job(
    input_dir: str,
    language: Optional[str],
    workers: int,
    enable_mask: bool,
    output_csv: Optional[str],
    output_dir: Optional[str],
    explicit_files: Optional[List[str]] = None,
    selected_files: Optional[List[str]] = None,
) -> None:
    with _job_lock:
        _job_state.update(
            {
                "status": "running",
                "progress": {"current": 0, "total": 0},
                "log": [],
                "csv_path": None,
                "params": {
                    "input_dir": input_dir,
                    "model": DEFAULT_MODEL,
                    "language": language,
                    "workers": workers,
                    "enable_mask": enable_mask,
                    "output_csv": output_csv,
                    "output_dir": output_dir,
                    "explicit_files": explicit_files,
                },
                "error": "",
                "started_at": time.time(),
                "ended_at": None,
            }
        )

    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("環境変数 OPENAI_API_KEY が設定されていません")
        client = OpenAI(api_key=api_key)

        # アップロード経由は explicit_files を最優先（探索の失敗で 0件になるのを防ぐ）
        if explicit_files:
            files = [Path(p) for p in explicit_files]
            # 安全のため最終フィルタはかける
            files = [p for p in files if _is_supported_audio_path(p)]
            _log(f"アップロード経由: 受領ファイル {len(explicit_files)} / 音声候補 {len(files)}")
        else:
            files = discover_audio_files(input_dir)

        if selected_files:
            wanted = set(selected_files)
            files_before = len(files)
            files = [p for p in files if p.name in wanted or str(p.relative_to(input_dir)) in wanted]
            _log(f"UI選択フィルタ: {files_before} -> {len(files)}")

        if not files:
            raise FileNotFoundError("処理対象の音声ファイルが見つかりませんでした")

        with _job_lock:
            _job_state["progress"]["total"] = len(files)

        _log(f"対象ファイル数: {len(files)}")
        _log(f"モデル: {DEFAULT_MODEL}")
        _log(f"言語: {language if language else '自動検出'}")
        _log(f"並列数: {workers}")
        _log(f"マスク: {'有効' if enable_mask else '無効'}")

        # 出力先の決定
        csv_path: Path
        if output_csv:
            csv_path = Path(output_csv)
        elif output_dir:
            csv_path = Path(output_dir) / "transcripts.csv"
        else:
            csv_path = Path(input_dir) / "transcripts.csv"
        _log(f"保存先CSVパス: {csv_path}")

        order_map = {str(p): idx for idx, p in enumerate(files)}
        results: List[Dict[str, Any]] = []

        mask_available = enable_mask and mask_text and _MASK_TOKENS

        def transcribe_file(audio_path: Path) -> Dict[str, Any]:
            start = time.time()
            compressed_path = None
            target_path = audio_path
            
            # 25MB 超過チェック
            try:
                size_bytes = audio_path.stat().st_size
                _log(f"ファイルサイズ確認: {audio_path.name} = {size_bytes / 1024 / 1024:.2f} MB")
                
                if size_bytes > 25 * 1024 * 1024:
                    _log(f"サイズ超過({size_bytes / 1024 / 1024:.1f}MB): 圧縮中... {audio_path.name}")
                    try:
                        compressed_path = _compress_audio(audio_path)
                        target_path = compressed_path
                        new_size = target_path.stat().st_size
                        _log(f"圧縮完了: {new_size / 1024 / 1024:.1f}MB")
                    except Exception as e:
                        msg = f"圧縮失敗: {e}"
                        if not _check_ffmpeg():
                            msg += " (ffmpegがインストールされていない可能性があります)"
                        
                        # 25MB超で圧縮失敗なら、APIに送っても失敗確定なので中断する
                        if size_bytes > 25 * 1024 * 1024:
                            raise ValueError(f"ファイルサイズ({size_bytes / 1024 / 1024:.1f}MB)が25MBを超えており、圧縮も失敗したため中断しました。\nヒント: ffmpegをインストールしてください。\n詳細: {msg}")

                        _log(f"{msg} -> オリジナルを使用します")
            except Exception as e:
                # statエラーなどは無視して進む
                pass

            try:
                with open(target_path, "rb") as f:
                    params = {"model": DEFAULT_MODEL, "file": f}
                    if language:
                        params["language"] = language
                    resp = client.audio.transcriptions.create(**params)
                text = getattr(resp, "text", "")
                masked_text = mask_text(text, _MASK_TOKENS) if mask_available else text
                return {
                    "file": str(audio_path),
                    "success": True,
                    "text": text,
                    "text_masked": masked_text,
                    "error": "",
                    "model": DEFAULT_MODEL,
                    "language": language,
                    "process_time_sec": time.time() - start,
                }
            except Exception as e:
                return {
                    "file": str(audio_path),
                    "success": False,
                    "text": "",
                    "text_masked": "",
                    "error": str(e),
                    "model": DEFAULT_MODEL,
                    "language": language,
                    "process_time_sec": time.time() - start,
                }
            finally:
                # 圧縮ファイルを削除
                if compressed_path and compressed_path.exists():
                    try:
                        compressed_path.unlink()
                    except Exception:
                        pass

        if workers <= 1:
            for p in files:
                r = transcribe_file(p)
                status_str = "OK" if r["success"] else "NG"
                msg = f"{status_str}: {p.name} ({r['process_time_sec']:.2f}s)"
                if not r["success"]:
                    msg += f" -> {r['error']}"
                _log(msg)
                results.append(r)
                with _job_lock:
                    _job_state["progress"]["current"] += 1
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(transcribe_file, p): p for p in files}
                for fut in as_completed(futures):
                    r = fut.result()
                    name = Path(r.get("file", "")).name or "?"
                    status_str = "OK" if r["success"] else "NG"
                    msg = f"{status_str}: {name} ({r['process_time_sec']:.2f}s)"
                    if not r["success"]:
                        msg += f" -> {r['error']}"
                    _log(msg)
                    results.append(r)
                    with _job_lock:
                        _job_state["progress"]["current"] += 1

        # 元の順序で並び替えてCSV書き込み
        results.sort(key=lambda x: order_map.get(x.get("file", ""), 0))
        write_csv(results, csv_path)

        ok = [r for r in results if r.get("success")]
        ng = [r for r in results if not r.get("success")]
        total_time = time.time() - _job_state["started_at"]
        _log(f"完了: 成功 {len(ok)} / 失敗 {len(ng)} / 合計 {len(files)} / 時間 {total_time:.2f}s")
        _log(f"CSV: {csv_path}")

        with _job_lock:
            _job_state.update(
                {
                    "status": "done",
                    "csv_path": str(csv_path),
                    "ended_at": time.time(),
                }
            )
    except Exception as e:
        _log(f"エラー: {e}")
        with _job_lock:
            _job_state.update({"status": "error", "error": str(e), "ended_at": time.time()})


@app.route("/", methods=["GET"])
def index() -> str:
    return """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <title>Whisper 管理画面</title>
  <style>
    body { font-family: sans-serif; margin: 24px; }
    label { display: block; margin-top: 12px; }
    input[type=text], input[type=number] { width: 360px; padding: 6px; }
    button { margin-top: 16px; padding: 10px 16px; }
    #log { white-space: pre-wrap; background: #f7f7f7; padding: 12px; border: 1px solid #ddd; height: 260px; overflow: auto; }
  </style>
</head>
<body>
  <h2>Whisper 一括文字起こし</h2>
  <div>
    <h3>フォルダをアップロードして実行</h3>
    <ol>
      <li>フォルダを選択（サブフォルダも可）</li>
      <li>対象ファイルをチェック（不要なものは外す）</li>
      <li>出力先（任意）を選択して実行</li>
    </ol>
    <input id="input_files" type="file" webkitdirectory multiple />
    <div style="margin:8px 0;">
      <button onclick="checkAllUpload()">全選択</button>
      <button onclick="uncheckAllUpload()">全解除</button>
    </div>
    <div id="upload_files_box" style="margin-top:8px; max-height:220px; overflow:auto; border:1px solid #ddd; padding:8px;"></div>
    <p>モデル: whisper-1 (固定)</p>
    <label>言語コード (空なら自動): <input id="language" type="text" placeholder="ja or en" /></label>
    <label>並列数 (workers): <input id="workers" type="number" value="4" min="1" /></label>
    <div>
      <label>出力フォルダ (任意):</label>
      <div style="margin-top:4px;">
        <select id="output_dir_select">
          <option value="">入力フォルダ/transcripts.csv（デフォルト）</option>
          <option value="downloads">~/Downloads/（transcripts.csv を作成）</option>
          <option value="documents">~/Documents/（transcripts.csv を作成）</option>
          <option value="tmp">/tmp/（transcripts.csv を作成）</option>
          <option value="custom">カスタム指定</option>
        </select>
        <input id="output_dir_custom" type="text" placeholder="/absolute/path/to/output_folder" style="width:360px; margin-left:8px;" disabled />
      </div>
    </div>
    <label><input id="enable_mask" type="checkbox" /> マスクを有効化</label>
    <div style="margin-top:8px;">
      <button onclick="startJob()">実行</button>
    </div>
  </div>

  <h3>進捗</h3>
  <div id="progress">-</div>
  <div id="csv_link"></div>
  <h3>ログ</h3>
  <div id="log"></div>

  <script>
    let pollTimer = null;
    let uploadFiles = {};
    let uploadChecked = new Set();
    let seq = 0;
    const nextId = (prefix) => prefix + (++seq);
    const outputDirOption = () => {
      const select = document.getElementById('output_dir_select');
      const custom = document.getElementById('output_dir_custom');
      const val = select.value;
      if (val === 'custom') return custom.value.trim();
      if (val === 'downloads') return '~/Downloads';
      if (val === 'documents') return '~/Documents';
      if (val === 'tmp') return '/tmp';
      return '';
    };

    document.getElementById('output_dir_select').addEventListener('change', (e) => {
      const custom = document.getElementById('output_dir_custom');
      const enableCustom = e.target.value === 'custom';
      custom.disabled = !enableCustom;
      if (!enableCustom) custom.value = '';
      custom.focus();
    });

    function startJob() {
      const filesInput = document.getElementById('input_files');
      const hasFiles = filesInput && filesInput.files && filesInput.files.length > 0;
      if (!hasFiles) {
        alert('フォルダをアップロードしてください');
        return;
      }
      startUpload(filesInput.files);
    }

    function startUpload(files) {
      const fd = new FormData();
      let added = 0;
      
      // デバッグ: どんなファイル名で送ろうとしているか確認
      console.log("--- Upload Start ---");
      
      uploadChecked.forEach(name => {
        const file = uploadFiles[name];
        if (file) {
          // file.name だけだと階層構造が消えることがある。
          // webkitRelativePath があればそれを使うが、
          // なければ name を使う。
          // 重要: FormData に append する第3引数が filename としてサーバーに渡る
          const filename = file.webkitRelativePath || file.name;
          fd.append('files', file, filename);
          console.log(`Added: ${filename} (${file.size} bytes)`);
          added += 1;
        }
      });
      console.log(`Total files: ${added}`);

      if (!added) {
        alert('実行対象のファイルを選択してください');
        return;
      }
      fd.append('language', document.getElementById('language').value || '');
      fd.append('workers', document.getElementById('workers').value || '4');
      fd.append('output_dir', outputDirOption());
      fd.append('enable_mask', document.getElementById('enable_mask').checked ? '1' : '0');
      
      fetch('/upload', {
        method: 'POST',
        body: fd
      }).then(r => r.json()).then(data => {
        console.log("Response:", data);
        if (data.error) { alert(data.error); return; }
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(fetchStatus, 1500);
      }).catch(e => {
        console.error(e);
        alert(e);
      });
    }

    const filesInput = document.getElementById('input_files');
    filesInput.addEventListener('change', (e) => {
      const files = e.target.files || [];
      const box = document.getElementById('upload_files_box');
      box.innerHTML = '';
      uploadFiles = {};
      uploadChecked = new Set();
      if (!files.length) {
        box.textContent = 'フォルダを選択してください';
        return;
      }
      Array.from(files).forEach(f => {
        const key = f.webkitRelativePath || f.name;
        uploadFiles[key] = f;
        uploadChecked.add(key);
      });
      Object.keys(uploadFiles).forEach(name => {
        const id = nextId('up_');
        const div = document.createElement('div');
        div.innerHTML = `<label><input type="checkbox" id="${id}" checked data-name="${name}"> ${name}</label>`;
        box.appendChild(div);
        div.querySelector('input').addEventListener('change', (ev) => {
          if (ev.target.checked) uploadChecked.add(name); else uploadChecked.delete(name);
        });
      });
    });

    function checkAllUpload() {
      const box = document.getElementById('upload_files_box');
      const checks = box.querySelectorAll('input[type=checkbox]');
      checks.forEach(cb => { cb.checked = true; uploadChecked.add(cb.dataset.name); });
    }
    function uncheckAllUpload() {
      const box = document.getElementById('upload_files_box');
      const checks = box.querySelectorAll('input[type=checkbox]');
      checks.forEach(cb => { cb.checked = false; uploadChecked.delete(cb.dataset.name); });
    }

    function fetchStatus() {
      fetch('/status').then(r => r.json()).then(data => {
        const { status, progress, log, csv_path } = data;
        document.getElementById('progress').textContent =
          `状態: ${status} / ${progress.current} / ${progress.total}`;
        document.getElementById('log').textContent = log.join('\\n');
        if (status === 'done' && csv_path) {
          document.getElementById('csv_link').innerHTML =
            `<a href="/download">結果CSVをダウンロード</a>`;
          clearInterval(pollTimer);
        } else if (status === 'error') {
          document.getElementById('csv_link').textContent = '';
          clearInterval(pollTimer);
        }
      });
    }
    fetchStatus();
  </script>
</body>
</html>"""


@app.route("/start", methods=["POST"])
def start_job():
    data = request.get_json(silent=True) or request.form
    input_dir = (data.get("input_dir") or "").strip()
    language = (data.get("language") or None) or None
    workers = int(data.get("workers") or 4)
    output_csv = _normalize_user_path((data.get("output_csv") or "").strip())
    output_dir = _normalize_user_path((data.get("output_dir") or "").strip())
    enable_mask = bool(data.get("enable_mask") in [True, "true", "1", "on", 1])
    selected_files = data.get("selected_files") or None
    if isinstance(selected_files, list):
        selected_files = [str(x) for x in selected_files if x]

    if not input_dir:
        return jsonify({"error": "入力フォルダを指定してください"}), 400
    if workers < 1:
        workers = 1

    with _job_lock:
        if _job_state["status"] == "running":
            return jsonify({"error": "実行中のジョブがあります。完了をお待ちください"}), 409
        _job_state["id"] = str(uuid.uuid4())

    t = threading.Thread(
        target=_run_job,
        args=(input_dir, language, workers, enable_mask, output_csv, output_dir, None, selected_files),
        daemon=True,
    )
    t.start()
    return jsonify({"job_id": _job_state["id"], "status": "started"})


@app.route("/upload", methods=["POST"])
def upload_job():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "フォルダ（ファイル）をアップロードしてください"}), 400

    language = (request.form.get("language") or None) or None
    workers = int(request.form.get("workers") or 4)
    output_csv = _normalize_user_path((request.form.get("output_csv") or "").strip())
    output_dir = _normalize_user_path((request.form.get("output_dir") or "").strip())
    enable_mask = bool(request.form.get("enable_mask") in [True, "true", "1", "on", 1])

    if workers < 1:
        workers = 1

    with _job_lock:
        if _job_state["status"] == "running":
            return jsonify({"error": "実行中のジョブがあります。完了をお待ちください"}), 409
        _job_state["id"] = str(uuid.uuid4())

    temp_dir = Path(tempfile.mkdtemp(prefix="whisper_upload_"))
    try:
        saved = _save_uploaded_files(files, temp_dir)
    except Exception as e:
        return jsonify({"error": f"アップロード保存に失敗しました: {e}"}), 400

    if not saved:
        return jsonify({"error": "保存されたファイルがありません"}), 400

    # アップロードされた内容から、音声候補を確定（ここで0件なら詳細を返す）
    audio_candidates: List[str] = []
    debug_rows: List[Dict[str, Any]] = []
    for p, orig, mime, size in saved:
        ok = _is_supported_audio_path(p)
        debug_rows.append(
            {
                "orig": orig,
                "saved": str(p),
                "suffix": p.suffix,
                "mime": mime,
                "size": size,
                "accepted": ok,
            }
        )
        if ok:
            audio_candidates.append(str(p))

    if not audio_candidates:
        # 受け取ったファイル情報を返す（最大50件）
        return (
            jsonify(
                {
                    "error": "音声ファイルが見つかりませんでした（拡張子が未対応の可能性）",
                    "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
                    "received_files": debug_rows[:50],
                }
            ),
            400,
        )

    t = threading.Thread(
        target=_run_job,
        args=(str(temp_dir), language, workers, enable_mask, output_csv, output_dir, audio_candidates, None),
        daemon=True,
    )
    t.start()
    return jsonify(
        {
            "job_id": _job_state["id"],
            "status": "started",
            "tmp_dir": str(temp_dir),
            "audio_candidates": len(audio_candidates),
        }
    )




@app.route("/status", methods=["GET"])
def status():
    with _job_lock:
        data = {
            "id": _job_state.get("id"),
            "status": _job_state.get("status"),
            "progress": _job_state.get("progress"),
            "log": _job_state.get("log", []),
            "csv_path": _job_state.get("csv_path"),
            "error": _job_state.get("error"),
            "params": _job_state.get("params"),
            "started_at": _job_state.get("started_at"),
            "ended_at": _job_state.get("ended_at"),
        }
    return jsonify(data)


@app.route("/download", methods=["GET"])
def download_csv():
    with _job_lock:
        csv_path = _job_state.get("csv_path")
        status = _job_state.get("status")
    if status != "done" or not csv_path:
        return jsonify({"error": "CSVはまだ利用できません"}), 404
    path = Path(csv_path)
    if not path.exists():
        return jsonify({"error": "CSVファイルが見つかりません"}), 404
    return send_file(path, as_attachment=True, download_name=path.name)


if __name__ == "__main__":
    if not _check_ffmpeg():
        print("警告: ffmpegが見つかりません。25MBを超えるファイルの自動圧縮機能が動作しません。", file=sys.stderr)
        print("ヒント: pip install imageio-ffmpeg を実行するか、システムにffmpegをインストールしてください。", file=sys.stderr)

    host = os.environ.get("WHISPER_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("WHISPER_UI_PORT", "8000"))
    app.run(host=host, port=port, debug=False, threaded=True)

