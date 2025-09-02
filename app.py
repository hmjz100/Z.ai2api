# -*- coding: utf-8 -*-
"""
Z.ai → OpenAI-Compatible 代理 (Function Call + UTF-8 + SSE)
- /v1/chat/completions 与 /v1/models
- 严格兼容 OpenAI 工具调用：有 tool_calls 时 content=null、finish_reason=tool_calls
- SSE 流式：携带 tools 时全程缓冲，结束后统一输出（避免客户端崩溃/乱码）
- 明确 UTF-8：所有响应带 charset=utf-8；SSE 手动按 UTF-8 解码
"""

import os
import json
import re
import time
import logging
from typing import List, Dict, Any, Optional, Iterable

import requests
from flask import Flask, request, Response, jsonify, make_response

# ==============================
# 配置（环境变量优先）
# ==============================
API_BASE = os.getenv("ZAI_API_BASE", "https://chat.z.ai")
DEFAULT_PORT = int(os.getenv("PORT", "8080"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "GLM-4.5")
DEBUG_MODE = os.getenv("DEBUG", "true").lower() == "true"

# 思考链模式: think | pure | raw
THINK_TAGS_MODE = os.getenv("THINK_TAGS_MODE", "think")

# 工具调用开关
FUNCTION_CALL_ENABLED = os.getenv("FUNCTION_CALL_ENABLED", "true").lower() == "true"
ANON_TOKEN_ENABLED = os.getenv("ANON_TOKEN_ENABLED", "true").lower() == "true"

# 匿名失败时兜底用的上游 token（可为空）
UPSTREAM_TOKEN = os.getenv("ZAI_UPSTREAM_TOKEN", "").strip()

# 超时&重试
HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "10"))
HTTP_READ_TIMEOUT = float(os.getenv("HTTP_READ_TIMEOUT", "60"))
TOKEN_TIMEOUT = float(os.getenv("TOKEN_TIMEOUT", "8"))
RETRY_COUNT = int(os.getenv("RETRY_COUNT", "2"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "0.6"))

# 其他
MAX_JSON_SCAN = int(os.getenv("MAX_JSON_SCAN", "200000"))
SSE_HEARTBEAT_SECONDS = float(os.getenv("SSE_HEARTBEAT_SECONDS", "15"))

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/139.0.0.0",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "X-FE-Version": "prod-fe-1.0.76",
    "sec-ch-ua": '"Not;A=Brand";v="99", "Edge";v="139"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Origin": API_BASE,
}

# ==============================
# 日志
# ==============================
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("zai2openai")

def debug(msg, *args):
    if DEBUG_MODE:
        log.debug(msg, *args)

# ==============================
# Flask
# ==============================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # JSON 保持 UTF-8

# ==============================
# 通用函数
# ==============================
def set_cors(resp: Response) -> Response:
    resp.headers.update({
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    })
    return resp

def preflight() -> Response:
    return set_cors(make_response("", 204))

def now_ns_id(prefix="msg") -> str:
    return f"{prefix}-{int(time.time() * 1e9)}"

def _safe_log_json(prefix: str, data: Any):
    try:
        scrub = json.loads(json.dumps(data))
        for k in ("Authorization", "authorization"):
            if isinstance(scrub, dict) and k in scrub:
                scrub[k] = "***"
        debug("%s %s", prefix, json.dumps(scrub, ensure_ascii=False)[:2000])
    except Exception:
        debug("%s <unserializable>", prefix)

def get_requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    return s

def get_token(session: requests.Session) -> str:
    if not ANON_TOKEN_ENABLED:
        return UPSTREAM_TOKEN
    url = f"{API_BASE}/api/v1/auths/"
    for i in range(RETRY_COUNT + 1):
        try:
            r = session.get(url, timeout=TOKEN_TIMEOUT)
            r.raise_for_status()
            token = r.json().get("token")
            if token:
                debug("匿名 token 获取成功 (前10位): %s...", token[:10])
                return token
        except Exception as e:
            debug("匿名 token 获取失败[%s/%s]: %s", i + 1, RETRY_COUNT + 1, e)
            if i < RETRY_COUNT:
                time.sleep(RETRY_BACKOFF * (i + 1))
    return UPSTREAM_TOKEN  # 可能为空，依上游策略

def call_upstream_chat(session: requests.Session, data: Dict[str, Any], chat_id: str) -> requests.Response:
    headers = {
        **BROWSER_HEADERS,
        "Authorization": f"Bearer {get_token(session)}",
        "Referer": f"{API_BASE}/c/{chat_id}",
    }
    _safe_log_json("上游请求体:", data)
    url = f"{API_BASE}/api/chat/completions"
    for i in range(RETRY_COUNT + 1):
        try:
            r = session.post(
                url, json=data, headers=headers, stream=True,
                timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT)
            )
            if r.status_code // 100 != 2:
                raise RuntimeError(f"Upstream HTTP {r.status_code}: {r.text[:2000]}")
            return r
        except Exception as e:
            log.warning("上游调用失败[%s/%s]: %s", i + 1, RETRY_COUNT + 1, e)
            if i < RETRY_COUNT:
                time.sleep(RETRY_BACKOFF * (i + 1))
            else:
                raise
    raise RuntimeError("上游调用失败（重试耗尽）")

def parse_upstream_sse(upstream: requests.Response) -> Iterable[Dict[str, Any]]:
    """按 UTF-8 解码 Z.ai 的 SSE 行"""
    for raw in upstream.iter_lines(decode_unicode=False):
        if not raw:
            continue
        if isinstance(raw, bytes):
            s = raw.decode("utf-8", "ignore")
        else:
            s = raw
        if not isinstance(s, str) or not s.startswith("data: "):
            continue
        try:
            yield json.loads(s[6:])
        except Exception:
            continue

# ==============================
# 工具调用：注入 & 提取
# ==============================
def format_tools_for_prompt(tools: List[Dict]) -> str:
    if not tools:
        return ""
    lines = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fdef = tool.get("function", {}) or {}
        name = fdef.get("name", "unknown")
        desc = fdef.get("description", "")
        params = fdef.get("parameters", {}) or {}
        tool_desc = [f"- {name}: {desc}"]
        props = params.get("properties", {}) or {}
        required = set(params.get("required", []) or [])
        for pname, pinfo in props.items():
            ptype = (pinfo or {}).get("type", "any")
            pdesc = (pinfo or {}).get("description", "")
            req = " (required)" if pname in required else " (optional)"
            tool_desc.append(f"  - {pname} ({ptype}){req}: {pdesc}")
        lines.append("\n".join(tool_desc))
    if not lines:
        return ""
    return (
        "\n\n可用的工具函数:\n" + "\n".join(lines) +
        "\n\n如果需要调用工具，请仅用以下 JSON 结构回复（不要包含多余文本）:\n"
        "```json\n"
        "{\n"
        '  "tool_calls": [\n'
        "    {\n"
        '      "id": "call_xxx",\n'
        '      "type": "function",\n'
        '      "function": {\n'
        '        "name": "function_name",\n'
        '        "arguments": "{\\\"param1\\\": \\\"value1\\\"}"\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
    )

def _content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return " ".join(parts)
    return ""

def _append_text_to_content(orig: Any, extra: str) -> Any:
    if isinstance(orig, str):
        return orig + extra
    if isinstance(orig, list):
        newc = list(orig)
        if newc and isinstance(newc[-1], dict) and newc[-1].get("type") == "text":
            newc[-1]["text"] = newc[-1].get("text", "") + extra
        else:
            newc.append({"type": "text", "text": extra})
        return newc
    return extra

def process_messages_with_tools(messages: List[Dict],
                                tools: Optional[List[Dict]] = None,
                                tool_choice: Optional[Any] = None) -> List[Dict]:
    processed: List[Dict] = []
    if tools and FUNCTION_CALL_ENABLED and (tool_choice != "none"):
        tools_prompt = format_tools_for_prompt(tools)
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            for m in messages:
                if m.get("role") == "system":
                    mm = dict(m)
                    mm["content"] = _append_text_to_content(m.get("content"), tools_prompt)
                    processed.append(mm)
                else:
                    processed.append(m)
        else:
            processed = [{"role": "system", "content": "你是一个有用的助手。" + tools_prompt}] + messages

        if tool_choice in ("required", "auto"):
            if processed and processed[-1].get("role") == "user":
                last = dict(processed[-1])
                last["content"] = _append_text_to_content(last.get("content"), "\n\n请根据需要使用提供的工具函数。")
                processed[-1] = last
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fname = (tool_choice.get("function") or {}).get("name")
            if fname and processed and processed[-1].get("role") == "user":
                last = dict(processed[-1])
                last["content"] = _append_text_to_content(last.get("content"), f"\n\n请使用 {fname} 函数来处理这个请求。")
                processed[-1] = last
    else:
        processed = list(messages)

    final_msgs: List[Dict[str, Any]] = []
    for m in processed:
        role = m.get("role")
        if role in ("tool", "function"):
            tool_name = m.get("name", "unknown")
            tool_content = _content_to_str(m.get("content", ""))
            final_msgs.append({
                "role": "assistant",
                "content": f"工具 {tool_name} 返回结果:\n```json\n{tool_content}\n```",
            })
        else:
            mm = dict(m)
            if isinstance(mm.get("content"), list):
                mm["content"] = _content_to_str(mm["content"])
            final_msgs.append(mm)
    return final_msgs

# ==============================
# 思考链清洗
# ==============================
history_phase = "thinking"

def process_content_by_phase(content: str, phase: str) -> str:
    global history_phase
    raw = content

    if content and (phase in ("thinking", "answer") or "summary>" in content):
        content = re.sub(r"(?s)<details[^>]*?>.*?</details>", "", content)
        content = content.replace("</thinking>", "").replace("<Full>", "").replace("</Full>", "")

        if THINK_TAGS_MODE == "think":
            if phase == "thinking":
                content = content.lstrip("> ").replace("\n>", "\n").strip()
            content = re.sub(r"\n?<summary>.*?</summary>\n?", "", content, flags=re.DOTALL)
            content = re.sub(r"<details[^>]*>\n?", "<think>\n\n", content)
            content = re.sub(r"\n?</details>", "\n\n</think>", content)
            if phase == "answer":
                m = re.search(r"(?s)^(.*?</think>)(.*)$", content)
                if m:
                    before, after = m.groups()
                    if after.strip():
                        if history_phase == "thinking":
                            content = f"\n\n</think>\n\n{after.lstrip()}"
                        elif history_phase == "answer":
                            content = ""
                    else:
                        content = "\n\n</think>"

        elif THINK_TAGS_MODE == "pure":
            if phase == "thinking":
                content = re.sub(r"\n?<summary>.*?</summary>", "", content, flags=re.DOTALL)
            content = re.sub(r"<details[^>]*>\n?", "<details type=\"reasoning\">", content)
            content = re.sub(r"\n?</details>", "\n\n></details>", content)
            if phase == "answer":
                m = re.search(r"(?s)^(.*?</details>)(.*)$", content)
                if m:
                    _, after = m.groups()
                    if after.strip():
                        if history_phase == "thinking":
                            content = f"\n\n{after.lstrip()}"
                        elif history_phase == "answer":
                            content = ""
                    else:
                        content = ""
            content = re.sub(r"</?details[^>]*>", "", content)

        elif THINK_TAGS_MODE == "raw":
            if phase == "thinking":
                content = re.sub(r"\n?<summary>.*?</summary>", "", content, flags=re.DOTALL)
            content = re.sub(r"<details[^>]*>\n?", "<details type=\"reasoning\" open><div>\n\n", content)
            content = re.sub(r"\n?</details>", "\n\n</div></details>", content)
            if phase == "answer":
                m = re.search(r"(?s)^(.*?</details>)(.*)$", content)
                if m:
                    before, after = m.groups()
                    if after.strip():
                        if history_phase == "thinking":
                            content = f"\n\n</details>\n\n{after.lstrip()}"
                        elif history_phase == "answer":
                            content = ""
                    else:
                        dm = re.search(r'duration="(\d+)"', before)
                        sm = re.search(r"(?s)<summary>.*?</summary>", before)
                        if sm:
                            content = f"\n\n</div>{sm.group()}</details>\n\n"
                        elif dm:
                            content = f'\n\n</div><summary>Thought for {dm.group(1)} seconds</summary></details>\n\n'
                        else:
                            content = "\n\n</div></details>"

    history_phase = phase
    return content

def extract_content_from_sse(data: Dict[str, Any]) -> str:
    d = data.get("data", {}) or {}
    phase = d.get("phase")
    delta = d.get("delta_content", "") or ""
    edit = d.get("edit_content", "") or ""
    content = delta or edit
    if content and phase in ("answer", "thinking"):
        return process_content_by_phase(content, phase) or ""
    return content or ""

# ==============================
# 工具调用提取
# ==============================
_JSON_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_INLINE = re.compile(r"(\{[^{}]{0,10000}\"tool_calls\".*?\})", re.DOTALL)
_FUNC_LINE = re.compile(r"调用函数\s*[：:]\s*([\w\-\.]+)\s*(?:参数|arguments)[：:]\s*(\{.*?\})", re.DOTALL)

def try_extract_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None
    sample = text[:MAX_JSON_SCAN]

    fences = _JSON_FENCE.findall(sample)
    for js in fences:
        try:
            data = json.loads(js)
            if "tool_calls" in data and isinstance(data["tool_calls"], list):
                return data["tool_calls"]
        except Exception:
            continue

    m = _JSON_INLINE.search(sample)
    if m:
        js = m.group(1)
        try:
            data = json.loads(js)
            if "tool_calls" in data and isinstance(data["tool_calls"], list):
                return data["tool_calls"]
        except Exception:
            pass

    m2 = _FUNC_LINE.search(sample)
    if m2:
        fname = m2.group(1).strip()
        args = m2.group(2).strip()
        try:
            json.loads(args)
            return [{
                "id": now_ns_id("call"),
                "type": "function",
                "function": {"name": fname, "arguments": args},
            }]
        except Exception:
            return None

    return None

def strip_tool_json_from_text(text: str) -> str:
    def _drop_if_toolcalls(match: re.Match) -> str:
        block = match.group(1)
        try:
            data = json.loads(block)
            if "tool_calls" in data:
                return ""
        except Exception:
            pass
        return match.group(0)

    new_text = _JSON_FENCE.sub(_drop_if_toolcalls, text)
    new_text = _JSON_INLINE.sub("", new_text)
    return new_text.strip()

# ==============================
# 路由
# ==============================
@app.route("/healthz", methods=["GET"])
def healthz():
    resp = make_response("ok", 200)
    resp.headers["Content-Type"] = "text/plain; charset=utf-8"
    return set_cors(resp)

@app.route("/v1/models", methods=["GET", "OPTIONS"])
def models():
    if request.method == "OPTIONS":
        return preflight()
    session = get_requests_session()
    try:
        headers = {**BROWSER_HEADERS, "Authorization": f"Bearer {get_token(session)}"}
        r = session.get(f"{API_BASE}/api/models", headers=headers, timeout=HTTP_READ_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
        items = payload.get("data", []) or []

        def format_model_name(model_id: str, name: Optional[str]) -> str:
            def is_alpha(ch: str) -> bool:
                return ("A" <= ch <= "Z") or ("a" <= ch <= "z")
            if name and len(name) > 0 and is_alpha(name[0]):
                return name
            if not model_id:
                return name or "UNKNOWN"
            parts = model_id.split("-")
            if len(parts) == 1:
                return parts[0].upper()
            out = [parts[0].upper()]
            for p in parts[1:]:
                if not p:
                    out.append("")
                elif p.isdigit():
                    out.append(p)
                elif any(c.isalpha() for c in p):
                    out.append(p.capitalize())
                else:
                    out.append(p)
            return "-".join(out)

        models_out = []
        now_ts = int(time.time())
        for m in items:
            info = m.get("info", {}) or {}
            if not info.get("is_active", True):
                continue
            mid = m.get("id")
            mname = format_model_name(mid, m.get("name"))
            models_out.append({
                "id": mid,
                "object": "model",
                "name": mname,
                "created": info.get("created_at", now_ts),
                "owned_by": "z.ai",
            })

        response = set_cors(jsonify({"object": "list", "data": models_out}))
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response
    except Exception as e:
        log.exception("获取模型列表失败: %s", e)
        err = set_cors(jsonify({"error": {"message": "fetch models failed"}}))
        err.headers["Content-Type"] = "application/json; charset=utf-8"
        err.status_code = 500
        return err

@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return preflight()

    req = request.get_json(force=True, silent=True) or {}
    chat_id = now_ns_id("chat")
    msg_id = now_ns_id("msg")
    model = req.get("model") or DEFAULT_MODEL

    tools = req.get("tools", [])
    tool_choice = req.get("tool_choice")

    processed_messages = process_messages_with_tools(req.get("messages", []), tools, tool_choice)

    upstream_data = {
        "stream": bool(req.get("stream", False)),
        "chat_id": chat_id,
        "id": msg_id,
        "model": model,
        "messages": processed_messages,
        "features": {"enable_thinking": True},
        **{k: v for k, v in req.items() if k in ("temperature", "top_p", "max_tokens")},
    }

    session = get_requests_session()
    try:
        upstream = call_upstream_chat(session, upstream_data, chat_id)
    except Exception as e:
        resp = make_response(f"上游调用失败: {e}", 502)
        resp.headers["Content-Type"] = "text/plain; charset=utf-8"
        return set_cors(resp)

    created_ts = int(time.time())

    # --------- 流式 ----------
    if req.get("stream"):
        def event_stream() -> Iterable[str]:
            last_ping = time.time()
            acc_content = ""
            tool_calls: Optional[List[Dict[str, Any]]] = None
            buffering_only = FUNCTION_CALL_ENABLED and bool(tools)

            # 首块：role
            first_chunk = {
                "id": now_ns_id("chatcmpl"),
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}}],
            }
            yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

            for data in parse_upstream_sse(upstream):
                if time.time() - last_ping >= SSE_HEARTBEAT_SECONDS:
                    yield ": keep-alive\n\n"
                    last_ping = time.time()

                if (data.get("data") or {}).get("done"):
                    if buffering_only:
                        tool_calls = try_extract_tool_calls(acc_content)
                        if tool_calls:
                            out = {
                                "id": now_ns_id("chatcmpl"),
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": model,
                                "choices": [{"index": 0, "delta": {"tool_calls": []}}],
                            }
                            for i, tc in enumerate(tool_calls):
                                out["choices"][0]["delta"]["tool_calls"].append({
                                    "index": i,
                                    "id": tc.get("id"),
                                    "type": tc.get("type", "function"),
                                    "function": tc.get("function", {}),
                                })
                            yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
                            finish = "tool_calls"
                        else:
                            trimmed = strip_tool_json_from_text(acc_content)
                            if trimmed:
                                chunk = {
                                    "id": now_ns_id("chatcmpl"),
                                    "object": "chat.completion.chunk",
                                    "created": created_ts,
                                    "model": model,
                                    "choices": [{"index": 0, "delta": {"content": trimmed}}],
                                }
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            finish = "stop"
                    else:
                        finish = "stop"

                    tail = {
                        "id": now_ns_id("chatcmpl"),
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    }
                    yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                piece = extract_content_from_sse(data)
                if not piece:
                    continue

                if buffering_only:
                    acc_content += piece  # 工具模式：全程缓冲
                else:
                    chunk = {
                        "id": now_ns_id("chatcmpl"),
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": piece}}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        resp = Response(event_stream())
        resp.headers["Content-Type"] = "text/event-stream; charset=utf-8"
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Connection"] = "keep-alive"
        resp.headers["X-Accel-Buffering"] = "no"
        return set_cors(resp)

    # --------- 非流式 ----------
    full_text = ""
    for d in parse_upstream_sse(upstream):
        full_text += extract_content_from_sse(d)

    tool_calls = None
    finish_reason = "stop"
    if FUNCTION_CALL_ENABLED and tools:
        tool_calls = try_extract_tool_calls(full_text)
        if tool_calls:
            # content 必须为 null（OpenAI 规范）
            full_text = strip_tool_json_from_text(full_text)
            finish_reason = "tool_calls"

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": None if tool_calls else (full_text or ""),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    resp_body = {
        "id": now_ns_id("chatcmpl"),
        "object": "chat.completion",
        "created": created_ts,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    response = set_cors(jsonify(resp_body))
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

# ==============================
# 主入口
# ==============================
if __name__ == "__main__":
    log.info(
        "代理启动: port=%s, default_model=%s, think_mode=%s, func_call=%s, debug=%s",
        DEFAULT_PORT, DEFAULT_MODEL, THINK_TAGS_MODE, FUNCTION_CALL_ENABLED, DEBUG_MODE
    )
    app.run(host="0.0.0.0", port=DEFAULT_PORT, threaded=True)
