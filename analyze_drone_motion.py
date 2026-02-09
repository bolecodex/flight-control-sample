#!/usr/bin/env python3
import asyncio
import base64
import gzip
import json
import mimetypes
import os
import struct
import subprocess
import sys
import tempfile
import uuid
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import websockets


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def image_to_data_url(image_path: Path) -> str:
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def extract_output_text(response_json: dict) -> str:
    output = response_json.get("output", [])
    texts = []
    for item in output:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                texts.append(part.get("text", ""))
    return "\n".join(t for t in texts if t)


def extract_stream_delta(event_json: dict) -> str:
    if "delta" in event_json and isinstance(event_json.get("delta"), str):
        return event_json.get("delta", "")
    output = event_json.get("output", [])
    for item in output:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if "delta" in part and isinstance(part.get("delta"), str):
                return part.get("delta", "")
    content = event_json.get("content", [])
    for part in content:
        if isinstance(part, dict) and "delta" in part and isinstance(part.get("delta"), str):
            return part.get("delta", "")
    return ""


def extract_stream_text(event_json: dict) -> str:
    if "text" in event_json and isinstance(event_json.get("text"), str):
        return event_json.get("text", "")
    output = event_json.get("output", [])
    texts = []
    for item in output:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
                texts.append(part.get("text", ""))
    content = event_json.get("content", [])
    for part in content:
        if isinstance(part, dict) and part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
            texts.append(part.get("text", ""))
    return "".join(texts)

def parse_cli_args(argv: List[str]) -> Tuple[Optional[str], Optional[Path], List[Path]]:
    text_input = None
    audio_input = None
    image_args = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--text":
            if i + 1 < len(argv):
                text_input = argv[i + 1]
            skip_next = True
            continue
        if arg == "--audio":
            if i + 1 < len(argv):
                audio_input = Path(argv[i + 1]).resolve()
            skip_next = True
            continue
        if arg == "--serve":
            continue
        if arg == "--port":
            skip_next = True
            continue
        if arg.startswith("--"):
            continue
        image_args.append(Path(arg).resolve())
    return text_input, audio_input, image_args


def build_asr_header(message_type: int, flags: int, serialization: int, compression: int) -> bytes:
    version = 0b0001
    header_size = 0b0001
    header = (version << 12) | (header_size << 8) | (message_type << 4) | flags
    header = (header << 8) | (serialization << 4) | compression
    header = (header << 8)
    return header.to_bytes(4, "big")


def gzip_payload(payload: bytes) -> bytes:
    return gzip.compress(payload)


def read_wav_pcm(audio_path: Path) -> tuple[bytes, int, int, int]:
    if audio_path.suffix.lower() == ".wav":
        with wave.open(str(audio_path), "rb") as wf:
            channels = wf.getnchannels()
            rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            bits = sampwidth * 8
            data = wf.readframes(wf.getnframes())
        return data, rate, bits, channels
    data = audio_path.read_bytes()
    rate = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
    bits = int(os.getenv("ASR_BITS", "16"))
    channels = int(os.getenv("ASR_CHANNELS", "1"))
    return data, rate, bits, channels

def read_wav_bytes(audio_path: Path) -> bytes:
    return audio_path.read_bytes()

def parse_wav_info(data: bytes) -> tuple[int, int, int, int]:
    if len(data) < 44:
        raise ValueError("Invalid WAV data")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Invalid WAV header")
    num_channels = struct.unpack("<H", data[22:24])[0]
    sample_rate = struct.unpack("<I", data[24:28])[0]
    bits_per_sample = struct.unpack("<H", data[34:36])[0]
    pos = 36
    data_size = 0
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if chunk_id == b"data":
            data_size = chunk_size
            break
        pos += 8 + chunk_size
    if data_size == 0:
        raise ValueError("Invalid WAV data chunk")
    return num_channels, bits_per_sample, sample_rate, data_size


def chunk_audio(data: bytes, rate: int, bits: int, channels: int, chunk_ms: int = 200) -> list[bytes]:
    bytes_per_sample = bits // 8
    bytes_per_second = rate * channels * bytes_per_sample
    chunk_size = max(1, bytes_per_second * chunk_ms // 1000)
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def extract_asr_text(payload: dict) -> str:
    if not payload:
        return ""
    if isinstance(payload.get("text"), str) and payload.get("text"):
        return payload.get("text", "")
    results = payload.get("result")
    if isinstance(results, list):
        for item in reversed(results):
            if isinstance(item, dict):
                text_value = item.get("text") or item.get("sentence") or item.get("utterance")
                if isinstance(text_value, str) and text_value:
                    return text_value
    utterances = payload.get("utterances")
    if isinstance(utterances, list):
        for item in reversed(utterances):
            if isinstance(item, dict):
                text_value = item.get("text") or item.get("utterance") or item.get("sentence")
                if isinstance(text_value, str) and text_value:
                    return text_value
    return ""

def parse_asr_message(message: bytes) -> Optional[dict]:
    if len(message) < 4:
        return None
    header_size = message[0] & 0x0F
    message_type = message[1] >> 4
    flags = message[1] & 0x0F
    serialization_method = message[2] >> 4
    compression = message[2] & 0x0F
    payload = message[header_size * 4:]
    if flags & 0x01:
        if len(payload) < 4:
            return None
        payload = payload[4:]
    if flags & 0x04:
        if len(payload) < 4:
            return None
        payload = payload[4:]
    if message_type == 0b1001:
        if len(payload) < 4:
            return None
        payload_size = struct.unpack(">I", payload[:4])[0]
        payload = payload[4:4 + payload_size]
    elif message_type == 0b1111:
        if len(payload) < 8:
            return None
        payload_size = struct.unpack(">I", payload[4:8])[0]
        payload = payload[8:8 + payload_size]
    if not payload:
        return None
    if compression == 0b0001:
        try:
            payload = gzip.decompress(payload)
        except OSError:
            return None
    if serialization_method == 0b0001:
        try:
            return json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            return None
    return None


async def run_asr(audio_path: Path, on_partial: Optional[Callable[[str], None]] = None) -> str:
    app_key = os.getenv("ASR_APP_ID") or os.getenv("ASR_APP_KEY", "")
    access_key = os.getenv("ASR_ACCESS_TOKEN") or os.getenv("ASR_ACCESS_KEY", "")
    resource_id = os.getenv("ASR_RESOURCE_ID", "volc.seedasr.sauc.duration")
    if not app_key or not access_key or not resource_id:
        raise RuntimeError("未找到 ASR_APP_ID/ASR_ACCESS_TOKEN/ASR_RESOURCE_ID，请检查 .env 或环境变量。")

    ws_url = os.getenv(
        "ASR_WS_URL",
        "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
    )
    connect_id = str(uuid.uuid4())
    headers = {
        "X-Api-App-Key": app_key,
        "X-Api-Access-Key": access_key,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Connect-Id": connect_id,
    }

    wav_bytes = read_wav_bytes(audio_path)
    channels, bits, rate, data_size = parse_wav_info(wav_bytes)
    bytes_per_sec = channels * (bits // 8) * rate
    segment_bytes = max(bytes_per_sec * 200 // 1000, 1)
    chunks = [wav_bytes[i:i + segment_bytes] for i in range(0, len(wav_bytes), segment_bytes)]
    request_payload = {
        "user": {"uid": "flight-control-sample"},
        "audio": {
            "format": "wav",
            "codec": "raw",
            "rate": rate,
            "bits": bits,
            "channel": channels,
        },
        "request": {
            "model_name": "bigmodel",
            "enable_punc": True,
            "enable_itn": True,
            "show_utterances": True,
            "enable_nonstream": False,
        },
    }
    request_bytes = gzip_payload(json.dumps(request_payload, ensure_ascii=False).encode("utf-8"))
    header = build_asr_header(message_type=0b0001, flags=0b0001, serialization=0b0001, compression=0b0001)
    sequence = 1
    first_packet = header + struct.pack(">i", sequence) + struct.pack(">I", len(request_bytes)) + request_bytes

    transcript = ""
    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        await ws.send(first_packet)
        sequence += 1
        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            flags = 0b0011 if is_last else 0b0001
            audio_header = build_asr_header(message_type=0b0010, flags=flags, serialization=0b0000, compression=0b0001)
            audio_payload = gzip_payload(chunk)
            seq_value = -sequence if is_last else sequence
            audio_packet = audio_header + struct.pack(">i", seq_value) + struct.pack(">I", len(audio_payload)) + audio_payload
            await ws.send(audio_packet)
            sequence += 1
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                continue
            if isinstance(response, bytes) and len(response) >= 8:
                payload = parse_asr_message(response)
                text = extract_asr_text(payload or {})
                if text:
                    transcript = text
                    if on_partial:
                        on_partial(text)
        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=2)
                if isinstance(response, bytes) and len(response) >= 8:
                    payload = parse_asr_message(response)
                    text = extract_asr_text(payload or {})
                    if text:
                        transcript = text
                        if on_partial:
                            on_partial(text)
        except asyncio.TimeoutError:
            pass
    return transcript


def request_text_control(text_input: str, base_url: str, headers: dict, model: str) -> str:
    prompt = (
        "你是无人机飞控规划助手。用户将通过文本描述意图，请结合运镜美学与无人机分控技术，"
        "给出下一秒飞控指导建议。只输出JSON，包含："
        "t（整数秒，表示下一秒时刻），"
        "v_mps（速度，m/s，数字，正值代表前进，负值代表后退），"
        "yaw_rate_dps（偏航角速度，度/秒，数字，正值顺时针，负值逆时针），"
        "pitch_deg（云台俯仰角，度，数字，向下为负），"
        "stabilization（一句话稳定与平滑建议）。"
    )
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": "只根据文本指令给出下一秒飞控建议。"},
            {"role": "user", "content": [{"type": "input_text", "text": prompt}, {"type": "input_text", "text": text_input}]},
        ],
        "max_output_tokens": 256,
        "temperature": 0.3,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    url = base_url.rstrip("/") + "/responses"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
    response_json = json.loads(body)
    return extract_output_text(response_json) or json.dumps(response_json, ensure_ascii=False)

def normalize_base64(data: str) -> str:
    cleaned = data.strip()
    if "base64," in cleaned:
        cleaned = cleaned.split("base64,", 1)[1]
    return cleaned.replace("\n", "").replace("\r", "")

def convert_to_wav(input_path: Path, output_path: Path, rate: int, channels: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ar",
            str(rate),
            "-ac",
            str(channels),
            "-f",
            "wav",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def create_handler(base_dir: Path, base_url: str, headers: dict, model: str):
    class ControlHandler(BaseHTTPRequestHandler):
        server_version = "FlightControlServer/1.0"

        def send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def do_GET(self) -> None:
            raw_path = self.path.split("?", 1)[0]
            request_path = "/index.html" if raw_path == "/" else raw_path
            target = (base_dir / request_path.lstrip("/")).resolve()
            if not str(target).startswith(str(base_dir.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not target.exists() or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type, _ = mimetypes.guess_type(str(target))
            data = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            if self.path == "/api/health":
                self.send_json(HTTPStatus.OK, {"status": "ok"})
                return
            payload = self.read_json()
            if self.path == "/api/text":
                text_input = str(payload.get("text", "")).strip()
                if not text_input:
                    self.send_json(HTTPStatus.BAD_REQUEST, {"error": "请输入文本指令"})
                    return
                try:
                    result = request_text_control(text_input, base_url, headers, model)
                except HTTPError as e:
                    error_body = e.read().decode("utf-8") if e.fp else ""
                    self.send_json(e.code, {"error": "请求失败", "detail": error_body})
                    return
                self.send_json(HTTPStatus.OK, {"text": text_input, "result": result})
                return
            if self.path == "/api/voice/stream":
                audio_b64 = payload.get("audio_base64")
                mime_type = payload.get("mime_type", "audio/webm")
                if not audio_b64:
                    self.send_json(HTTPStatus.BAD_REQUEST, {"error": "未收到音频数据"})
                    return
                extension = mimetypes.guess_extension(mime_type) or ".webm"
                base64_text = normalize_base64(str(audio_b64))
                try:
                    audio_bytes = base64.b64decode(base64_text)
                except base64.binascii.Error:
                    self.send_json(HTTPStatus.BAD_REQUEST, {"error": "音频数据无法解析"})
                    return
                if not audio_bytes:
                    self.send_json(HTTPStatus.BAD_REQUEST, {"error": "音频为空，请重新录制"})
                    return
                rate = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
                channels = int(os.getenv("ASR_CHANNELS", "1"))
                input_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
                wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                input_path = Path(input_file.name)
                wav_path = Path(wav_file.name)
                input_file.close()
                wav_file.close()
                try:
                    input_path.write_bytes(audio_bytes)
                    if extension != ".wav":
                        convert_to_wav(input_path, wav_path, rate, channels)
                    else:
                        wav_path.write_bytes(audio_bytes)
                    if not wav_path.exists() or wav_path.stat().st_size == 0:
                        raise RuntimeError("音频转换失败")
                except Exception:
                    input_path.unlink(missing_ok=True)
                    wav_path.unlink(missing_ok=True)
                    self.send_json(HTTPStatus.BAD_REQUEST, {"error": "音频格式处理失败"})
                    return
                    self.send_json(HTTPStatus.BAD_REQUEST, {"error": "音频格式处理失败"})
                    return

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                def send_event(event: str, data: dict) -> None:
                    message = f"event: {event}\n" + f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    self.wfile.write(message.encode("utf-8"))
                    self.wfile.flush()

                async def handle_stream() -> None:
                    try:
                        def on_partial(text: str) -> None:
                            send_event("transcript", {"text": text})

                        text_input = await run_asr(wav_path, on_partial=on_partial)
                    except RuntimeError as exc:
                        send_event("error", {"error": str(exc)})
                        return
                    if not text_input:
                        send_event("error", {"error": "未识别到有效文本"})
                        return
                    try:
                        result = request_text_control(text_input, base_url, headers, model)
                    except HTTPError as e:
                        error_body = e.read().decode("utf-8") if e.fp else ""
                        send_event("error", {"error": "请求失败", "detail": error_body})
                        return
                    send_event("result", {"text": text_input, "result": result})

                try:
                    asyncio.run(handle_stream())
                finally:
                    input_path.unlink(missing_ok=True)
                    wav_path.unlink(missing_ok=True)
                return
            if self.path == "/api/voice":
                audio_b64 = payload.get("audio_base64")
                mime_type = payload.get("mime_type", "audio/webm")
                if not audio_b64:
                    self.send_json(HTTPStatus.BAD_REQUEST, {"error": "未收到音频数据"})
                    return
                extension = mimetypes.guess_extension(mime_type) or ".webm"
                base64_text = normalize_base64(str(audio_b64))
                try:
                    audio_bytes = base64.b64decode(base64_text)
                except base64.binascii.Error:
                    self.send_json(HTTPStatus.BAD_REQUEST, {"error": "音频数据无法解析"})
                    return
                if not audio_bytes:
                    self.send_json(HTTPStatus.BAD_REQUEST, {"error": "音频为空，请重新录制"})
                    return
                rate = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
                channels = int(os.getenv("ASR_CHANNELS", "1"))
                input_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
                wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                input_path = Path(input_file.name)
                wav_path = Path(wav_file.name)
                input_file.close()
                wav_file.close()
                try:
                    input_path.write_bytes(audio_bytes)
                    if extension != ".wav":
                        convert_to_wav(input_path, wav_path, rate, channels)
                    else:
                        wav_path.write_bytes(audio_bytes)
                    try:
                        text_input = asyncio.run(run_asr(wav_path))
                    except RuntimeError as exc:
                        self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    if not text_input:
                        self.send_json(HTTPStatus.BAD_REQUEST, {"error": "未识别到有效文本"})
                        return
                    try:
                        result = request_text_control(text_input, base_url, headers, model)
                    except HTTPError as e:
                        error_body = e.read().decode("utf-8") if e.fp else ""
                        self.send_json(e.code, {"error": "请求失败", "detail": error_body})
                        return
                    self.send_json(HTTPStatus.OK, {"text": text_input, "result": result})
                    return
                finally:
                    input_path.unlink(missing_ok=True)
                    wav_path.unlink(missing_ok=True)
            self.send_error(HTTPStatus.NOT_FOUND)

    return ControlHandler

def run_server(port: int, base_dir: Path, base_url: str, headers: dict, model: str) -> None:
    handler = create_handler(base_dir, base_url, headers, model)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"服务已启动: http://localhost:{port}")
    server.serve_forever()

def main() -> int:
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / ".env"
    if not env_path.exists():
        env_path = script_dir.parent / ".env"
    load_env(env_path)

    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        print("未找到 ARK_API_KEY，请检查 .env 或环境变量。", file=sys.stderr)
        return 1

    if "--serve" in sys.argv:
        port = 8000
        if "--port" in sys.argv:
            index = sys.argv.index("--port")
            if index + 1 < len(sys.argv):
                try:
                    port = int(sys.argv[index + 1])
                except ValueError:
                    print("端口格式不正确，需为整数。", file=sys.stderr)
                    return 1
        base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        model = os.getenv("ARK_MODEL", "doubao-seed-1-8-251228")
        run_server(port, script_dir, base_url, headers, model)
        return 0

    text_input, audio_input, image_args = parse_cli_args(sys.argv[1:])
    if audio_input:
        if not audio_input.exists():
            print(f"未找到音频文件: {audio_input}", file=sys.stderr)
            return 1
    if not image_args and not text_input and not audio_input:
        candidate_dirs = [
            script_dir / "cliff-video" / "frames",
            script_dir / "frames",
        ]
        image_paths = []
        for candidate in candidate_dirs:
            if candidate.exists():
                image_paths = sorted(candidate.glob("*.png"))
                if image_paths:
                    break
        if not image_paths:
            print("未找到可用的图片文件，请指定图片路径或确保默认帧目录存在。", file=sys.stderr)
            return 1
    elif image_args:
        image_paths = image_args
    else:
        image_paths = []
    for image_path in image_paths:
        if not image_path.exists():
            print(f"未找到图片文件: {image_path}", file=sys.stderr)
            return 1

    model = os.getenv("ARK_MODEL", "doubao-seed-1-8-251228")
    window_size = int(os.getenv("WINDOW_SIZE", "5"))
    step_size = int(os.getenv("STEP_SIZE", "1"))
    has_sequence = len(image_paths) >= window_size

    base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if text_input or audio_input:
        if audio_input:
            try:
                text_input = asyncio.run(run_asr(audio_input))
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        if not text_input:
            print("未识别到有效文本，请检查音频内容或输入文本。", file=sys.stderr)
            return 1
        try:
            result = request_text_control(text_input, base_url, headers, model)
        except HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            print(f"请求失败: HTTP {e.code}", file=sys.stderr)
            if error_body:
                print(error_body, file=sys.stderr)
            return 1
        print(result)
        return 0

    url = base_url.rstrip("/") + "/responses"

    if has_sequence:
        prompt = (
            "你是无人机航拍运镜分析与规划助手。以下是从一个连续视频流中按时间先后顺序截取的5张连续帧画面，"
            "每帧间隔1秒。请基于这5帧的内容，并结合运镜美学、运镜技巧、无人机分控技术，给出下一秒的飞控指导建议。"
            "只输出JSON，包含："
            "t（全局时间戳，整数秒，表示下一秒时刻），"
            "v_mps（速度，m/s，数字，正值代表前进，负值代表后退），"
            "yaw_rate_dps（偏航角速度，度/秒，数字，正值顺时针，负值逆时针），"
            "pitch_deg（云台俯仰角，度，数字，向下为负），"
            "stabilization（一句话稳定与平滑建议）。"
            "若无法确定，请给出最合理的推测并标注不确定性。"
        )

        previous_response_id = None
        cache_enabled = True
        for start in range(0, len(image_paths) - window_size + 1, step_size):
            window = image_paths[start:start + window_size]
            content = [{"type": "input_text", "text": prompt}]
            for index, image_path in enumerate(window):
                content.append({"type": "input_text", "text": f"第{index + 1}帧图像"})
                content.append({"type": "input_image", "image_url": image_to_data_url(image_path)})

            payload = {
                "model": model,
                "input": [
                    {"role": "system", "content": "只根据图像给出无人机运镜建议。"},
                    {"role": "user", "content": content},
                ],
                "max_output_tokens": 256,
                "temperature": 0.3,
                "thinking": {"type": "disabled"},
            }
            if cache_enabled:
                payload["caching"] = {"type": "enabled"}
                if previous_response_id:
                    payload["previous_response_id"] = previous_response_id

            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = Request(url, data=data, headers=headers, method="POST")
            try:
                with urlopen(req, timeout=60) as resp:
                    body = resp.read().decode("utf-8")
            except HTTPError as e:
                error_body = e.read().decode("utf-8") if e.fp else ""
                if e.code == 403 and "AccessDenied.CacheService" in error_body:
                    cache_enabled = False
                    payload.pop("caching", None)
                    payload.pop("previous_response_id", None)
                    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    req = Request(url, data=data, headers=headers, method="POST")
                    try:
                        with urlopen(req, timeout=60) as resp:
                            body = resp.read().decode("utf-8")
                    except HTTPError as retry_error:
                        retry_body = retry_error.read().decode("utf-8") if retry_error.fp else ""
                        print(f"请求失败: HTTP {retry_error.code}", file=sys.stderr)
                        if retry_body:
                            print(retry_body, file=sys.stderr)
                        return 1
                elif e.code == 400 and "previous response" in error_body and "not cached" in error_body:
                    cache_enabled = False
                    payload.pop("caching", None)
                    payload.pop("previous_response_id", None)
                    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    req = Request(url, data=data, headers=headers, method="POST")
                    try:
                        with urlopen(req, timeout=60) as resp:
                            body = resp.read().decode("utf-8")
                    except HTTPError as retry_error:
                        retry_body = retry_error.read().decode("utf-8") if retry_error.fp else ""
                        print(f"请求失败: HTTP {retry_error.code}", file=sys.stderr)
                        if retry_body:
                            print(retry_body, file=sys.stderr)
                        return 1
                else:
                    print(f"请求失败: HTTP {e.code}", file=sys.stderr)
                    if error_body:
                        print(error_body, file=sys.stderr)
                        if "InvalidEndpointOrModel.NotFound" in error_body:
                            print("请将 ARK_MODEL 设置为你已开通的模型ID，或在命令行第二个参数传入模型ID。", file=sys.stderr)
                    return 1

            response_json = json.loads(body)
            if cache_enabled:
                previous_response_id = response_json.get("id")
            output_text = extract_output_text(response_json)
            window_label = f"{window[0].name}..{window[-1].name}"
            global_t = start + window_size
            if output_text:
                try:
                    parsed = json.loads(output_text)
                    if isinstance(parsed, dict):
                        parsed["t"] = global_t
                        print(f"{window_label} => {json.dumps(parsed, ensure_ascii=False)}")
                    else:
                        print(f"{window_label} => {output_text}")
                except json.JSONDecodeError:
                    print(f"{window_label} => {output_text}")
            else:
                print(f"{window_label} => {json.dumps(response_json, ensure_ascii=False)}")
        return 0

    prompt = (
        "你是无人机航拍运镜分析与规划助手。请基于输入图像判断画面主体与空间结构，"
        "给出无人机运动方式建议，目标是画面稳定、运镜流畅、观感高级。"
        "只输出JSON，包含以下字段："
        "recommended_motion（一句话概述），"
        "path（路径与方向，例如缓慢环绕/平移/推进/拉远），"
        "speed（速度建议），"
        "gimbal（云台俯仰与航向控制建议），"
        "stabilization（稳定性与平滑建议），"
        "alternatives（最多2条备选运镜）。"
        "若无法确定，请给出最合理的推测并标注不确定性。"
    )

    content = [{"type": "input_text", "text": prompt}]
    for index, image_path in enumerate(image_paths):
        content.append({"type": "input_text", "text": f"第{index}秒帧图像"})
        content.append({"type": "input_image", "image_url": image_to_data_url(image_path)})

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": "只根据图像给出无人机运镜建议。"},
            {"role": "user", "content": content},
        ],
        "max_output_tokens": 512,
        "temperature": 0.3,
        "thinking": {"type": "disabled"},
        "stream": False,
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=60) as resp:
            if payload.get("stream"):
                streamed_text = ""
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        print()
                        return 0
                    try:
                        event_json = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = extract_stream_delta(event_json)
                    if delta:
                        print(delta, end="", flush=True)
                        streamed_text += delta
                        continue
                    text = extract_stream_text(event_json)
                    if text:
                        if text.startswith(streamed_text):
                            new_part = text[len(streamed_text):]
                        else:
                            new_part = text
                        if new_part:
                            print(new_part, end="", flush=True)
                            streamed_text += new_part
                print()
                return 0
            body = resp.read().decode("utf-8")
    except HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        print(f"请求失败: HTTP {e.code}", file=sys.stderr)
        if error_body:
            print(error_body, file=sys.stderr)
            if "InvalidEndpointOrModel.NotFound" in error_body:
                print("请将 ARK_MODEL 设置为你已开通的模型ID，或在命令行第二个参数传入模型ID。", file=sys.stderr)
        return 1

    response_json = json.loads(body)
    output_text = extract_output_text(response_json)
    if output_text:
        print(output_text)
        return 0

    print(json.dumps(response_json, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
