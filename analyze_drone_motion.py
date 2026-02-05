#!/usr/bin/env python3
import base64
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


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

    image_args = [Path(p).resolve() for p in sys.argv[1:] if not p.startswith("--")]
    if not image_args:
        image_args = [script_dir / "frames" / "frame_000001.png"]
    image_paths = image_args
    for image_path in image_paths:
        if not image_path.exists():
            print(f"未找到图片文件: {image_path}", file=sys.stderr)
            return 1

    model = os.getenv("ARK_MODEL", "doubao-seed-1-8-251228")
    window_size = int(os.getenv("WINDOW_SIZE", "5"))
    step_size = int(os.getenv("STEP_SIZE", "1"))
    has_sequence = len(image_paths) >= window_size

    base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    url = base_url.rstrip("/") + "/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if has_sequence:
        prompt = (
            "你是无人机航拍运镜分析与规划助手。以下是按时间顺序排列的连续5帧图像，"
            "每帧间隔1秒。请基于这5帧的时序变化，仅预测第5帧时刻无人机的动作与云台控制。"
            "只输出JSON，包含："
            "t（全局时间戳，整数秒），"
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
            global_t = start + window_size - 1
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
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=60) as resp:
            if payload["stream"]:
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
