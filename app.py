# -*- coding: utf-8 -*-
"""
Z.ai 2 API
将 Z.ai 代理为 OpenAI/Anthropic Compatible 格式，支持免令牌、智能处理思考链、图片上传（仅登录后）等功能
基于 https://github.com/kbykb/OpenAI-Compatible-API-Proxy-for-Z 使用 AI 辅助重构。
"""

import os, json, re, requests, urllib.parse, logging, uuid, base64, hashlib, hmac, tzlocal
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from flask import Flask, request, Response, jsonify, make_response
from typing import Any, Dict, List, Union, Optional

from dotenv import load_dotenv
load_dotenv()

# 配置
PROTOCOL = str(os.getenv("PROTOCOL", "https:"))
BASE = str(os.getenv("BASE", "chat.z.ai"))
PORT = int(os.getenv("PORT", "8080"))
MODEL = str(os.getenv("MODEL", "GLM-4.5"))
TOKEN = str(os.getenv("TOKEN", "")).strip()
DEBUG_MODE = str(os.getenv("DEBUG", "false")).lower() == "true"
THINK_TAGS_MODE = str(os.getenv("THINK_TAGS_MODE", "reasoning"))
ANONYMOUS_MODE = str(os.getenv("ANONYMOUS_MODE", "true")).lower() == "true"

# tiktoken 预加载
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tiktoken') + os.sep
os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
assert os.path.exists(os.path.join(cache_dir, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4")) # cl100k_base.tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

BROWSER_HEADERS = {
	"Accept": "*/*",
	"Accept-Language": "zh-CN,zh;q=0.9",
	"Cache-Control": "no-cache",
	"Connection": "keep-alive",
	"Content-Type": "application/json",
	"Origin": f"{PROTOCOL}//{BASE}",
	"Pragma": "no-cache",
	"Referer": f"{PROTOCOL}//{BASE}/",
	"Sec-Ch-Ua": '"Microsoft Edge";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
	"Sec-Ch-Ua-Mobile": "?0",
	"Sec-Ch-Ua-Platform": '"Windows"',
	"Sec-Fetch-Dest": "empty",
	"Sec-Fetch-Mode": "cors",
	"Sec-Fetch-Site": "same-origin",
	"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0",
	"X-FE-Version": "prod-fe-1.0.95",
}

# 日志
logging.basicConfig(
	level=logging.DEBUG if DEBUG_MODE else logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)
temp = {}

# Flask 应用
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

phaseBak = "thinking"
# 工具函数
class utils:
	@staticmethod
	class request:
		@staticmethod
		def chat(data, chat_id):
			timestamp = int(datetime.now().timestamp() * 1000)
			requestId = str(uuid.uuid4())

			user = utils.request.user()
			userToken = user.get("token")
			userId = user.get("id")

			params = {
				"timestamp": timestamp,
				"requestId": requestId,
			}
			"""
				"version": "0.0.1",
				"platform": "web",
				"token": userToken,
				"user_agent": BROWSER_HEADERS.get("User-Agent"),
				"language": "zh-CN",
				"languages": "zh-CN;en;en-US",
				"timezone": tzlocal.get_localzone_name(),
				"cookie_enabled": True,
				"screen_width": "1920",
				"screen_height": "1080",
				"screen_resolution": "1920x1080",
				"viewport_height": "880",
				"viewport_width": "1286",
				"viewport_size": "1286x880",
				"color_depth": "24",
				"pixel_ratio": "1",
				"current_url": f"{PROTOCOL}//{BASE}/c/{chat_id}",
				"pathname": f"/c/{chat_id}",
				"search": f"/c/{chat_id}",
				"hash": f"/c/{chat_id}",
				"host": BASE,
				"hostname": BASE,
				"protocol": PROTOCOL,
				"referrer": None,
				"title": "Z.ai Chat - Free AI powered by GLM-4.6 & GLM-4.5",
				"timezone_offset": int(-datetime.now(ZoneInfo(tzlocal.get_localzone_name())).utcoffset().total_seconds() / 60), # pyright: ignore[reportOptionalMemberAccess]
				"local_time": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
				"utc_time": datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT"),
				"is_mobile": False,
				"is_touch": False,
				"max_touch_points": "10",
				"browser_name": "Chrome",
				"os_name": "Windows",
			"""
			headers = {
				**BROWSER_HEADERS,
				"Authorization": f"Bearer {userToken}",
				"Referer": f"{PROTOCOL}//{BASE}/c/{chat_id}"
			}

			if not ANONYMOUS_MODE and userId:
				# user 的最后一句话
				last_user_message = ""
				for msg in reversed(data.get("messages", [])):
					if msg.get("role") == "user":
						content = msg.get("content")
						if isinstance(content, str):
							last_user_message = content.strip()
						elif isinstance(content, list):
							texts = [item.get("text", "") for item in content if item.get("type") == "text"]
							last_user_message = "".join(texts).strip()
						break

				signatures = utils.request.signature(timestamp, requestId, userId, last_user_message)
				headers["X-Signature"] = signatures.get("signature")
				params["user_id"] = userId
				params["signature_timestamp"] = signatures.get("timestamp")

			log.debug("发送请求:")
			log.debug("  headers: %s", json.dumps(headers))
			log.debug("  data: %s", json.dumps(data))
			return requests.post(f"{PROTOCOL}//{BASE}/api/chat/completions?{urllib.parse.urlencode(params)}", json=data, headers=headers, stream=True)
		@staticmethod
		def image(data_url, chat_id):
			try:
				if ANONYMOUS_MODE or not data_url.startswith("data:"):
					return None

				header, encoded = data_url.split(",", 1)
				mime_type = header.split(";")[0].split(":")[1] if ":" in header else "image/jpeg"

				image_data = base64.b64decode(encoded) # 解码数据
				filename = str(uuid.uuid4())

				log.debug("上传文件：%s", filename)
				response = requests.post(f"{PROTOCOL}//{BASE}/api/v1/files/", files={"file": (filename, image_data, mime_type)}, headers={**BROWSER_HEADERS, "Authorization": f"Bearer {utils.request.user().get("token")}", "Referer": f"{PROTOCOL}//{BASE}/c/{chat_id}"})

				if response.status_code == 200:
					result = response.json()
					return f"{result.get("id")}_{result.get("filename")}"
				else:
					raise Exception(response.text)
			except Exception as e:
				log.error("图片上传失败: %s", e)
			return None
		@staticmethod
		def id(prefix = "msg") -> str:
			return f"{prefix}-{int(datetime.now().timestamp()*1e9)}"
		@staticmethod
		def user():
			headers = BROWSER_HEADERS.copy()
			current_token = None if ANONYMOUS_MODE else TOKEN

			# 1. 尝试从缓存中获取 id
			if current_token and "tokens" in temp and current_token in temp["tokens"]:
				cached_id = temp["tokens"][current_token]
				log.debug("从缓存获取用户信息: id=%s, token=%s...", cached_id, current_token[:15])
				return {"id": cached_id, "token": current_token}

			# 2. 缓存未命中，发起请求
			if not ANONYMOUS_MODE:
				headers["Authorization"] = f"Bearer {TOKEN}"

			try:
				r = requests.get(f"{PROTOCOL}//{BASE}/api/v1/auths/", headers=headers)
				r.raise_for_status()
				data = r.json()
				userId = data.get("id")
				userToken = data.get("token")

				if not userToken and not ANONYMOUS_MODE:
					userToken = TOKEN

				# 3. 写入缓存（仅当有有效 token 时）
				if userToken and userId:
					if "tokens" not in temp:
						temp["tokens"] = {}
					temp["tokens"][userToken] = userId

				log.debug("用户信息: id=%s, token=%s...", userId, userToken[:15] if userToken else None)
				return {"id": userId, "token": userToken}
			except Exception as e:
				log.error("获取用户信息失败: %s", e)
				fallback_token = TOKEN if not ANONYMOUS_MODE else None
				# 注意：失败时不缓存，避免缓存错误状态
				return {"id": None, "token": fallback_token}
		@staticmethod
		def signature(timestamp: int, requestId: str, userId: str, text: str):
			text = text.strip()

			def _hmac_sha256(key: bytes, msg: bytes):
				return hmac.new(key, msg, hashlib.sha256)

			# 当前毫秒时间戳
			signature_t = int(datetime.now().timestamp() * 1000)

			# 拼接签名字符串
			signature_i = f"requestId,{requestId},timestamp,{timestamp},user_id,{userId}|{text}|{signature_t}"

			# 第一次 HMAC（时间片作为字符串转字节）
			time_slice = str(signature_t // (5 * 60 * 1000)).encode("utf-8")
			signature_n = _hmac_sha256(b"junjie", time_slice).digest()

			# 第二次 HMAC（签名字符串转字节）
			signature = _hmac_sha256(signature_n, signature_i.encode("utf-8")).hexdigest()

			# 调试日志
			log.debug("生成签名：%s", signature)
			log.debug("  timestamp: %s", timestamp)
			log.debug("  timestamp_t: %s", signature_t)
			log.debug("  requestId: %s", requestId)
			log.debug("  userId: %s", userId)
			log.debug("  text: %s", text)

			return {
				"signature": signature,
				"timestamp": signature_t
			}

		@staticmethod
		def response(resp):
			resp.headers.update({
				"Access-Control-Allow-Origin": "*",
				"Access-Control-Allow-Methods": "GET, POST, OPTIONS",
				"Access-Control-Allow-Headers": "Content-Type, Authorization",
			})
			return resp
		@staticmethod
		def format(data: Dict[str, Any], type: str = "OpenAI") -> Dict[str, Any]:
			"""
			格式化输入数据，支持 OpenAI 和 Anthropic 格式。
			返回标准化后的 data，供内部 chat 接口使用。
			"""
			odata = data.copy()
			messages = []

			# === 1. 处理 system（仅 Anthropic）===
			if type == "Anthropic" and "system" in odata:
				system = odata["system"]
				if isinstance(system, str):
					content = system.lstrip('\n')
				else:  # list of text blocks
					content = "\n".join(
						s.get("text", "").lstrip('\n')
						for s in system
						if s.get("type") == "text"
					)
				messages.append({"role": "system", "content": content})

			# === 2. 预处理 messages ===
			raw_messages = odata.get("messages", [])
			chat_id = odata.get("chat_id")  # 用于 OpenAI 图片上传

			for msg in raw_messages:
				role = msg.get("role")
				content = msg.get("content", [])
				new_msg = {"role": role}

				# --- OpenAI: 上传 data URL 图片 ---
				if type == "OpenAI" and isinstance(content, list):
					for item in content:
						if item.get("type") == "image_url":
							url = item.get("image_url", {}).get("url", "")
							if url.startswith(""):
								try:
									uploaded_url = utils.request.image(url, chat_id)
									if uploaded_url:
										item["image_url"]["url"] = uploaded_url
								except Exception:
									pass  # 或记录日志

				# --- 标准化 content ---
				if isinstance(content, str):
					new_msg["content"] = content
					messages.append(new_msg)
					continue

				# content 是 list，统一处理
				is_tool_result_msg = (
					type == "Anthropic"
					and role == "user"
					and any(item.get("type") == "tool_result" for item in content)
				)

				if is_tool_result_msg:
					# Anthropic: tool_result → role: tool
					for item in content:
						if item.get("type") == "tool_result":
							tool_call_id = item.get("tool_use_id")
							tool_content = item.get("content", [])
							if isinstance(tool_content, list):
								text = "".join(t.get("text", "") for t in tool_content if t.get("type") == "text")
							else:
								text = str(tool_content)
							messages.append({
								"role": "tool",
								"tool_call_id": tool_call_id,
								"content": text
							})
					continue  # 跳过常规处理

				# --- 提取文本和非文本内容 ---
				text_parts = []
				other_parts = []
				tool_calls = []

				for item in content:
					item_type = item.get("type")
					if item_type == "text":
						text_parts.append(item.get("text", ""))
					elif type == "Anthropic" and role == "assistant" and item_type == "tool_use":
						# Anthropic assistant 的 tool_use
						tool_calls.append({
							"id": item.get("id"),
							"type": "function",
							"function": {
								"name": item.get("name"),
								"arguments": json.dumps(item.get("input", {}) or {}, ensure_ascii=False)
							}
						})
					else:
						# 图片或其他内容（OpenAI 或 Anthropic user/image）
						other_parts.append(item)

				# --- 构建 content ---
				if not other_parts:
					# 全是文本 → 合并为字符串
					new_msg["content"] = "".join(text_parts) if text_parts else ""
				else:
					# 混合内容 → 保留结构
					new_msg["content"] = []
					for t in text_parts:
						new_msg["content"].append({"type": "text", "text": t})
					for item in other_parts:
						if type == "Anthropic" and item.get("type") == "image":
							source = item.get("source", {})
							if source.get("type") == "base64":
								media_type = source.get("media_type", "image/jpeg")
								data_b64 = source.get("data", "")
								new_msg["content"].append({
									"type": "image_url",
									"image_url": {"url": f"{media_type};base64,{data_b64}"}
								})
						else:
							# OpenAI 的 image_url 或其他未知类型，直接透传
							new_msg["content"].append(item)

				# --- 添加 tool_calls（仅 Anthropic assistant）---
				if tool_calls:
					new_msg["tool_calls"] = tool_calls
					# 若无文本，content 设为 None（与原逻辑一致）
					if not text_parts:
						new_msg["content"] = None

				messages.append(new_msg)

			# === 3. 构建最终 data ===
			result = {
				**odata,
				"messages": messages,
				"stream": True,  # 始终设为 True（根据你原逻辑）
			}

			# features 处理
			if type == "Anthropic":
				thinking_enabled = str(odata.get("thinking", {}).get("type", "enabled")).lower() == "enabled"
				result["features"] = odata.get("features", {"enable_thinking": thinking_enabled})
			else:  # OpenAI
				result["features"] = odata.get("features", {"enable_thinking": True})

			return result
	@staticmethod
	class response:
		@staticmethod
		def parse(stream):
			for line in stream.iter_lines():
				if not line or not line.startswith(b"data: "): continue
				try: data = json.loads(line[6:].decode("utf-8", "ignore"))
				except: continue
				yield data
		@staticmethod
		def format(data, type = "OpenAI"):
			data = data.get("data", "")
			if not data: return None
			phase = data.get("phase", "other")
			content = data.get("delta_content") or data.get("edit_content") or ""
			if not content: return None
			contentBak = content
			global phaseBak

			if phase == "tool_call":
				content = re.sub(r"\n*<glm_block[^>]*>{\"type\": \"mcp\", \"data\": {\"metadata\": {", "{", content)
				content = re.sub(r"\", \"result\": \"\".*</glm_block>", "", content)
			elif phase == "other" and phaseBak == "tool_call" and "glm_block" in content:
				phase = "tool_call"
				content = re.sub(r"null, \"display_result\": \"\".*</glm_block>", "\"}", content)

			if phase == "thinking" or (phase == "answer" and "summary>" in content):
				content = re.sub(r"(?s)<details[^>]*?>.*?</details>", "", content)
				content = content.replace("</thinking>", "").replace("<Full>", "").replace("</Full>", "")

				if phase == "thinking":
					content = re.sub(r'\n*<summary>.*?</summary>\n*', '\n\n', content)

				# 以 <reasoning> 为基底
				content = re.sub(r"<details[^>]*>\n*", "<reasoning>\n\n", content)
				content = re.sub(r"\n*</details>", "\n\n</reasoning>", content)

				if phase == "answer":
					match = re.search(r"(?s)^(.*?</reasoning>)(.*)$", content) # 判断 </reasoning> 后是否有内容
					if match:
						before, after = match.groups()
						if after.strip():
							# </reasoning> 后有内容
							if phaseBak == "thinking":
								# 思考休止 → 结束思考，加上回答
								content = f"\n\n</reasoning>\n\n{after.lstrip('\n')}"
							elif phaseBak == "answer":
								# 回答休止 → 清除所有
								content = ""
						else:
							# 思考休止 → </reasoning> 后无内容
							content = "\n\n</reasoning>"

				if THINK_TAGS_MODE == "reasoning":
					if phase == "thinking": content = re.sub(r'\n>\s?', '\n', content)
					content = re.sub(r'\n*<summary>.*?</summary>\n*', '', content)
					content = re.sub(r"<reasoning>\n*", "", content)
					content = re.sub(r"\n*</reasoning>", "", content)
				elif THINK_TAGS_MODE == "think":
					if phase == "thinking": content = re.sub(r'\n>\s?', '\n', content)
					content = re.sub(r'\n*<summary>.*?</summary>\n*', '', content)
					content = re.sub(r"<reasoning>", "<think>", content)
					content = re.sub(r"</reasoning>", "</think>", content)
				elif THINK_TAGS_MODE == "strip":
					content = re.sub(r'\n*<summary>.*?</summary>\n*', '', content)
					content = re.sub(r"<reasoning>\n*", "", content)
					content = re.sub(r"</reasoning>", "", content)
				elif THINK_TAGS_MODE == "details":
					if phase == "thinking": content = re.sub(r'\n>\s?', '\n', content)
					content = re.sub(r"<reasoning>", "<details type=\"reasoning\" open><div>", content)
					thoughts = ""
					if phase == "answer":
						# 判断是否有 <summary> 内容
						summary_match = re.search(r"(?s)<summary>.*?</summary>", before)
						duration_match = re.search(r'duration="(\d+)"', before)
						if summary_match:
							# 有内容 → 直接照搬
							thoughts = f"\n\n{summary_match.group()}"
						# 判断是否有 duration 内容
						elif duration_match:
							# 有内容 → 通过 duration 生成 <summary>
							thoughts = f'\n\n<summary>Thought for {duration_match.group(1)} seconds</summary>'
					content = re.sub(r"</reasoning>", f"</div>{thoughts}</details>", content)
				else:
					content = re.sub(r"</reasoning>", "</reasoning>\n\n", content)
					log.debug("警告：THINK_TAGS_MODE 传入了未知的替换模式，将使用 <reasoning> 标签。")

			phaseBak = phase
			if repr(content) != repr(contentBak):
				log.debug("R 内容: %s %s", phase, repr(contentBak))
				log.debug("W 内容: %s %s", phase, repr(content))
			else:
				log.debug("R 内容: %s %s", phase, repr(contentBak))

			if phase == "thinking" and THINK_TAGS_MODE == "reasoning":
				if type == "Anthropic": return {"type": "thinking_delta", "thinking": content}
				return {"role": "assistant", "reasoning_content": content}
			if phase == "tool_call":
				return {"tool_call": content}
			elif repr(content):
				if type == "Anthropic": return {"type": "text_delta", "text": content}
				else: return {"role": "assistant", "content": content}
			else:
				return None
		@staticmethod
		def count(text):
			return len(enc.encode(text))

# 路由
@app.route("/v1/models", methods=["GET", "POST", "OPTIONS"])
def models():
	if request.method == "OPTIONS": return utils.request.response(make_response())
	try:
		def format_model_name(name: str) -> str:
			"""格式化模型名:
			- 单段: 全大写
			- 多段: 第一段全大写, 后续段首字母大写
			- 数字保持不变, 符号原样保留
			"""
			if not name: return ""
			parts = name.split('-')
			if len(parts) == 1:
				return parts[0].upper()
			formatted = [parts[0].upper()]
			for p in parts[1:]:
				if not p:
					formatted.append("")
				elif p.isdigit():
					formatted.append(p)
				elif any(c.isalpha() for c in p):
					formatted.append(p.capitalize())
				else:
					formatted.append(p)
			return "-".join(formatted)

		def is_english_letter(ch: str) -> bool:
			"""判断是否是英文字符 (A-Z / a-z)"""
			return 'A' <= ch <= 'Z' or 'a' <= ch <= 'z'

		headers = {**BROWSER_HEADERS, "Authorization": f"Bearer {utils.request.user().get("token")}"}
		r = requests.get(f"{PROTOCOL}//{BASE}/api/models", headers=headers).json()
		models = []
		for m in r.get("data", []):
			if not m.get("info", {}).get("is_active", True):
				continue
			model_id, model_name = m.get("id"), m.get("name")
			if model_id.startswith(("GLM", "Z")):
				model_name = model_id
			if not model_name or not is_english_letter(model_name[0]):
				model_name = format_model_name(model_id)
			models.append({
				"id": model_id,
				"object": "model",
				"name": model_name,
				"created": m.get("info", {}).get("created_at", int(datetime.now().timestamp() * 1000)),
				"owned_by": "z.ai"
			})
		return utils.request.response(jsonify({"object":"list","data":models}))
	except Exception as e:
		log.error("模型列表失败: %s", e)
		return utils.request.response(jsonify({"error":"fetch models failed"})), 500

@app.route("/v1/chat/completions", methods=["GET", "POST", "OPTIONS"])
def OpenAI_Compatible():
	if request.method == "OPTIONS": return utils.request.response(make_response())
	odata = request.get_json(force=True, silent=True) or {}

	id = utils.request.id("chat")
	model = odata.get("model", MODEL)
	messages = odata.get("messages", [])
	stream = odata.get("stream", False)
	include_usage = stream and odata.get("stream_options", {}).get("include_usage", False)

	odata = utils.request.format(odata, "OpenAI")

	data = {
		**odata,
		"chat_id": id,
		"id": utils.request.id(),
		"model": model
	}

	try:
		response = utils.request.chat(data, id)
		if hasattr(response, 'status_code') and response.status_code != 200:
			try:
				error_content = response.text[:500]
			except Exception:
				error_content = "无法读取响应内容"
			return utils.request.response(jsonify({
				"error": response.status_code,
				"message": error_content
			})), response.status_code
	except Exception as e:
		return utils.request.response(jsonify({
			"error": 502,
			"message": str(e)
		})), 502

	prompt_tokens = utils.response.count("".join(
		c if isinstance(c, str) else (c.get("text", "") if isinstance(c, dict) and c.get("type") == "text" else "")
		for m in messages
		for c in ([m["content"]] if isinstance(m.get("content"), str) else (m.get("content") or []))
	))
	if stream:
		def stream():
			completion_str = ""
			completion_tokens = 0

			# 处理流式响应数据
			for data in utils.response.parse(response):
				delta = utils.response.format(data, "OpenAI")

				if delta:
					yield f"data: {json.dumps({
						"id": utils.request.id('chatcmpl'),
						"object": "chat.completion.chunk",
						"created": int(datetime.now().timestamp() * 1000),
						"model": model,
						"choices": [
							{
								"index": 0,
								"delta": delta,
								"message": delta,
								"finish_reason": None
							}
						]
					})}\n\n"

					# 累积实际生成的内容
					if "content" in delta:
						completion_str += delta["content"]
						completion_tokens = utils.response.count(completion_str) # 计算 tokens
					if "reasoning_content" in delta:
						completion_str += delta["reasoning_content"]
						completion_tokens = utils.response.count(completion_str) # 计算 tokens
				else:
					continue

			yield f"data: {json.dumps({
				'id': utils.request.id('chatcmpl'),
				'object': 'chat.completion.chunk',
				'created': int(datetime.now().timestamp() * 1000),
				'model': model,
				'choices': [
					{
						'index': 0,
						'delta': {"role": "assistant"},
						'message': {"role": "assistant"},
						'finish_reason': "stop"
					}
				]
			})}\n\n"
			if include_usage:
				# 发送 usage 统计信息
				yield f"data: {json.dumps({
					"id": utils.request.id('chatcmpl'),
					"object": "chat.completion.chunk",
					"created": int(datetime.now().timestamp() * 1000),
					"model": model,
					"choices": [],
					"usage": {
						"prompt_tokens": prompt_tokens,
						"completion_tokens": completion_tokens,
						"total_tokens": prompt_tokens + completion_tokens
					}
				})}\n\n"

			# 发送 [DONE] 标志，表示流结束
			yield "data: [DONE]\n\n"

		# 返回 Flask 的流式响应
		return Response(stream(), mimetype="text/event-stream")
	else:
		# 上游不支持非流式，所以先用流式获取所有内容
		contents = {
			"content": [],
			"reasoning_content": []
		}
		for odata in utils.response.parse(response):
			if odata.get("data", {}).get("done"):
				break
			delta = utils.response.format(odata)
			if delta:
				if "content" in delta:
					contents["content"].append(delta["content"])
				if "reasoning_content" in delta:
					contents["reasoning_content"].append(delta["reasoning_content"])

		# 构建最终消息内容
		final_message = {"role": "assistant"}
		completion_str = ""
		if contents["reasoning_content"]:
			final_message["reasoning_content"] = "".join(contents["reasoning_content"])
			completion_str += "".join(contents["reasoning_content"])
		if contents["content"]:
			final_message["content"] = "".join(contents["content"])
			completion_str += "".join(contents["content"])
		completion_tokens = utils.response.count(completion_str) # 计算 tokens

		# 返回 Flask 响应
		return utils.request.response(jsonify({
			"id": utils.request.id("chatcmpl"),
			"object": "chat.completion",
			"created": int(datetime.now().timestamp() * 1000),
			"model": model,
			"choices": [{
				"index": 0,
				"delta": final_message,
				"message": final_message,
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": prompt_tokens,
				"completion_tokens": completion_tokens,
				"total_tokens": prompt_tokens + completion_tokens
			}
		}))

@app.route("/v1/messages", methods=["GET", "POST", "OPTIONS"])
def Anthropic_Compatible():
	if request.method == "OPTIONS": return utils.request.response(make_response())
	odata = request.get_json(force=True, silent=True) or {}

	id = utils.request.id("chat")
	model = odata.get("model", MODEL)
	stream = odata.get("stream", False)
	messages = []

	odata = utils.request.format(odata, "Anthropic")

	data = {
		**odata,
		"stream": True,
		"chat_id": id,
		"id": utils.request.id(),
		"model": model,
	}

	try:
		response = utils.request.chat(data, id)
		if hasattr(response, 'status_code') and response.status_code != 200:
			try:
				error_content = response.text[:500]
			except Exception:
				error_content = "无法读取响应内容"
			return utils.request.response(jsonify({
				"error": response.status_code,
				"message": error_content
			})), response.status_code
	except Exception as e:
		return utils.request.response(jsonify({
			"error": 502,
			"message": str(e)
		})), 502

	prompt_tokens = utils.response.count("".join(
		c if isinstance(c, str) else (c.get("text", "") if isinstance(c, dict) and c.get("type") == "text" else "")
		for m in messages
		for c in ([m["content"]] if isinstance(m.get("content"), str) else (m.get("content") or []))
	))
	if stream:
		def stream():
			completion_str = ""

			yield "event: message_start\n"
			yield f"data: {json.dumps({
				"type": "message_start",
				"message": {
					"id": utils.request.id(),
					"type": "message",
					"role": "assistant",
					"content": [],
					"model": model,
					"stop_reason": None,
					"stop_sequence": None,
					"usage": {
						"input_tokens": prompt_tokens,
						"output_tokens": 1
					}
				}
			})}\n\n"
			yield "event: content_block_start\n"
			yield f"data: {json.dumps({
				"type": "content_block_start",
				"index": 0,
				"content_block": {
					"type": "text",
					"text": ""
				}
			})}\n\n"
			yield "event: ping\n"
			yield f"data: {json.dumps({"type": "ping"})}\n\n"
			temp = {"tool_call": []}
			completion_tokens = 0
			# 处理流式响应数据
			for data in utils.response.parse(response):
				if data.get("data", {}).get("done"): break
				delta = utils.response.format(data, "Anthropic")
				call_stoped = False

				if delta:
					if "tool_call" in delta:
						temp["tool_call"].append(delta["tool_call"])
						# 尝试合并并解析，看是否构成完整 JSON
						tool_call_str = "".join(temp["tool_call"])
						try:
							tool_json = json.loads(tool_call_str)
							tool_imput = {}

							if "input" in tool_json:
								tool_imput = tool_json["input"]
								tool_json["input"] = {}

							if "arguments" in tool_json:
								try:
									tool_imput = json.dumps(json.loads(tool_json["arguments"]))
								except (json.JSONDecodeError, TypeError):
									log.warning("arguments 无法解析为 JSON，原值: %s", tool_json["arguments"])
								del tool_json["arguments"]

							log.debug("完整！调用！：%s", tool_json)
							call_stoped = True

							yield "event: content_block_stop\n"
							yield f"data: {json.dumps({"type": "content_block_stop", "index": 0})}\n\n"

							yield "event: content_block_start\n"
							yield f"data: {json.dumps({
								"type": "content_block_start",
								"index": 1,
								"content_block": {
									"type": "tool_use",
									**tool_json
								}
							})}\n\n"

							yield "event: content_block_delta\n"
							yield f"data: {json.dumps({
								"type": "content_block_delta",
								"index": 1,
								"delta": {
									"type": "input_json_delta",
									"partial_json": tool_imput
								}
							})}\n\n"

							yield "event: content_block_stop\n"
							yield f"data: {json.dumps({"type": "content_block_stop", "index": 1})}\n\n"

							break
						except json.JSONDecodeError:
							# JSON 不完整，继续收集
							continue
						except Exception as e:
							log.error(f"Tool call parse error: {e}")
							# 解析出其他错误，也应中断，避免 fallback
							return utils.request.response(make_response("Invalid tool call format", 500))
					if not call_stoped:
						yield "event: content_block_delta\n"
						yield f"data: {json.dumps({"type": "content_block_delta", "index": 0, "delta": delta})}\n\n"

					# 累积实际生成的内容
					if "text" in delta:
						completion_str += delta["text"]
						completion_tokens = utils.response.count(completion_str) # 计算 tokens
				else:
					continue

			yield "event: content_block_stop\n"
			yield f"data: {json.dumps({"type": "content_block_stop", "index": 0})}\n\n"
			# 发送 usage 统计信息
			yield "event: message_delta\n"
			yield f"data: {json.dumps({
				"type": "message_delta",
				"delta": {
					"stop_reason": "end_turn",
					"stop_sequence": None
				},
				"usage": {
					"output_tokens": completion_tokens
				}
			})}\n\n"
			yield "event: message_stop\n"
			yield f"data: {json.dumps({"type": "message_stop"})}\n\n"
			# 发送 [DONE] 标志，表示流结束
			yield "data: [DONE]\n\n"

		# 返回 Flask 的流式响应
		return Response(stream(), mimetype="text/event-stream")
	else:
		contents = {
			"content": [],
			"reasoning_content": [],
			"tool_call": []
		}
		has_tool_call = False  # 标记是否已确认 tool_call 完整

		for odata in utils.response.parse(response):
			if odata.get("data", {}).get("done"):
				break

			delta = utils.response.format(odata)
			if not delta:
				continue

			if "tool_call" in delta:
				contents["tool_call"].append(delta["tool_call"])
				# 尝试合并并解析，看是否构成完整 JSON
				tool_call_str = "".join(contents["tool_call"])
				try:
					tool_json = json.loads(tool_call_str)
					completion_tokens = utils.response.count("".join(contents["content"]))

					if "arguments" in tool_json:
						try:
							# 尝试将 arguments 解析为 JSON
							parsed_arguments = json.loads(tool_json["arguments"])
							# 替换为 input 字段并删除 arguments
							tool_json["input"] = parsed_arguments
							del tool_json["arguments"]
						except (json.JSONDecodeError, TypeError):
							# 如果解析失败，保留原始 arguments
							log.warning("arguments 无法解析为 JSON，保留原值: %s", tool_json["arguments"])

					log.debug("完整！调用！：%s", tool_json)
					return utils.request.response(jsonify({
						"id": utils.request.id(),
						"type": "message",
						"role": "assistant",
						"model": model,
						"content": [
							{
								"text": "".join(contents["content"]),
								"type": "text"
							} if contents["content"] else None,
							{
								"type": "tool_use",
								**tool_json
							}
						],
						"usage": {
							"input_tokens": prompt_tokens,
							"output_tokens": completion_tokens
						},
						"stop_sequence": None,
						"stop_reason": "tool_use",
					}))
				except json.JSONDecodeError:
					# JSON 不完整，继续收集
					continue
				except Exception as e:
					log.error(f"Tool call parse error: {e}")
					# 解析出其他错误，也应中断，避免 fallback
					return utils.request.response(make_response("Invalid tool call format", 500))

			# 只有没触发 tool_call 时，才继续收集文本
			if "content" in delta:
				contents["content"].append(delta["content"])
			if "reasoning_content" in delta:
				contents["reasoning_content"].append(delta["reasoning_content"])

		# === 循环结束，说明没有 tool_call 或 tool_call 未完整 ===

		# 构建纯文本响应
		completion_str = "".join(contents["reasoning_content"] + contents["content"])
		completion_tokens = utils.response.count(completion_str)

		return utils.request.response(jsonify({
			"id": utils.request.id(),
			"type": "message",
			"role": "assistant",
			"model": model,
			"content": [{
				"text": completion_str,
				"type": "text"
			}] if completion_str else [],
			"usage": {
				"input_tokens": prompt_tokens,
				"output_tokens": completion_tokens
			},
			"stop_sequence": None,
			"stop_reason": "end_turn",
		}))

# 健康检查
@app.route("/health")
def health():
	return utils.request.response(jsonify({
		"status": "ok",
		"timestamp": int(datetime.now().timestamp() * 1000)
	}))

# 主入口
if __name__ == "__main__":
	log.info("---------------------------------------------------------------------")
	log.info("Z.ai 2 API https://github.com/hmjz100/Z.ai2api")
	log.info("将 Z.ai 代理为 OpenAI/Anthropic Compatible 格式")
	log.info("基于 https://github.com/kbykb/OpenAI-Compatible-API-Proxy-for-Z 重构")
	log.info("---------------------------------------------------------------------")
	log.info(f"Base           {PROTOCOL}//{BASE}")
	log.info("Models         /v1/models")
	log.info("OpenAI         /v1/chat/completions")
	log.info("Anthropic      /v1/messages")
	log.info("---------------------------------------------------------------------")
	log.info("服务端口：%s", PORT)
	log.info("备选模型：%s", MODEL)
	log.info("思考处理：%s", THINK_TAGS_MODE)
	log.info("访客模式：%s", ANONYMOUS_MODE)
	log.info("显示调试：%s", DEBUG_MODE)
	app.run(host="0.0.0.0", port=PORT, threaded=True, debug=DEBUG_MODE)