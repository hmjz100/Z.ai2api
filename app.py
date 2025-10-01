# -*- coding: utf-8 -*-
"""
Z.ai 2 API
将 Z.ai 代理为 OpenAI/Anthropic Compatible 格式，支持免令牌、智能处理思考链、图片上传（仅登录后）等功能
基于 https://github.com/kbykb/OpenAI-Compatible-API-Proxy-for-Z 使用 AI 辅助重构。
"""

import os, json, re, requests, logging, uuid, base64
from datetime import datetime
from flask import Flask, request, Response, jsonify, make_response

from dotenv import load_dotenv
load_dotenv()

# 配置
BASE = str(os.getenv("BASE", "https://chat.z.ai"))
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
	"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0",
	"Accept": "*/*",
	"Accept-Language": "zh-CN,zh;q=0.9",
	"X-FE-Version": "prod-fe-1.0.88",
	"sec-ch-ua": '"Not;A=Brand";v="99", "Edge";v="139"',
	"sec-ch-ua-mobile": "?0",
	"sec-ch-ua-platform": '"Windows"',
	"Origin": BASE,
}

# 日志
logging.basicConfig(
	level=logging.DEBUG if DEBUG_MODE else logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

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
			log.debug("收到请求: %s", json.dumps(data))
			token = utils.request.token()
			return requests.post(f"{BASE}/api/chat/completions?timestamp={int(datetime.now().timestamp())}&platform=web", json=data, headers={**BROWSER_HEADERS, "Authorization": f"Bearer {token}", "Referer": f"{BASE}/c/{chat_id}"}, stream=True, timeout=60)
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
				response = requests.post(f"{BASE}/api/v1/files/", files={"file": (filename, image_data, mime_type)}, headers={**BROWSER_HEADERS, "Authorization": f"Bearer {utils.request.token()}", "Referer": f"{BASE}/c/{chat_id}"}, timeout=30)

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
		def token() -> str:
			if not ANONYMOUS_MODE: return TOKEN
			try:
				r = requests.get(f"{BASE}/api/v1/auths/", headers=BROWSER_HEADERS, timeout=8)
				token = r.json().get("token")
				if token:
					log.debug("获取匿名令牌: %s...", token[:15])
					return token
			except Exception as e:
				log.error("匿名令牌获取失败: %s", e)
			return TOKEN
		@staticmethod
		def response(resp):
			resp.headers.update({
				"Access-Control-Allow-Origin": "*",
				"Access-Control-Allow-Methods": "GET, POST, OPTIONS",
				"Access-Control-Allow-Headers": "Content-Type, Authorization",
			})
			return resp
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
				if type == "Anthropic": return {"type": "thinking_delta", "text": content}
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

		headers = {**BROWSER_HEADERS, "Authorization": f"Bearer {utils.request.token()}"}
		r = requests.get(f"{BASE}/api/models", headers=headers, timeout=8).json()
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
				"created": m.get("info", {}).get("created_at", int(datetime.now().timestamp())),
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
	features = odata.get("features", { "enable_thinking": True })
	stream = odata.get("stream", False)
	include_usage = stream and odata.get("stream_options", {}).get("include_usage", False)

	for message in messages:
		if isinstance(message.get("content"), list):
			for contentItem in message["content"]:
				if contentItem.get("type") == "image_url":
					url = contentItem.get("image_url", {}).get("url", "")
					if url.startswith("data:"):
						fileUrl = utils.request.image(url, id) # 上传图片
						if fileUrl:
							contentItem["image_url"]["url"] = fileUrl # 上传后的图片链接

	data = {
		**odata, 
		"stream": True,
		"chat_id": id,
		"id": utils.request.id(),
		"model": model,
		"messages": messages,
		"features": features
	}

	try:
		response = utils.request.chat(data, id)
	except Exception as e:
		return utils.request.response(make_response(f"上游请求失败: {e}", 502))

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
						"created": int(datetime.now().timestamp()),
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
				'created': int(datetime.now().timestamp()),
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
					"created": int(datetime.now().timestamp()),
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
			"created": int(datetime.now().timestamp()),
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
	omessages = odata.get("messages", [])
	stream = odata.get("stream", False)
	messages = []
	
	if "system" in odata:
		system_content = odata["system"]
		if isinstance(system_content, str):
			messages.append({"role": "system", "content": system_content.lstrip('\n')})
		elif isinstance(system_content, list):
			merged = "\n".join(s.get("text", "").lstrip('\n') for s in system_content if s.get("type") == "text")
			messages.append({"role": "system", "content": merged})

	for message in omessages:
		role = message.get("role")
		content = message.get("content", [])
		finalMessage = {
			"role": role,
			"content": []
		}

		if isinstance(content, str):
			finalMessage["content"] = content
			messages.append(finalMessage)
			continue

		# 判断是否是工具调用相关消息
		if role == "assistant":
			# 检查是否有 tool_use
			tool_calls = []
			text_parts = []

			for item in content:
				if item.get("type") == "text":
					text_parts.append(item.get("text", ""))
				elif item.get("type") == "tool_use":
					tool_call_id = item.get("id")
					tool_name = item.get("name")
					# 假设 tool_use 没有输入参数，或参数在 item 中
					tool_input = item.get("input", {})  # 可能是 dict 或 None

					tool_calls.append({
						"id": tool_call_id,
						"type": "function",
						"function": {
							"name": tool_name,
							"arguments": json.dumps(tool_input, ensure_ascii=False) if tool_input else "{}"
						}
					})

			# 合并文本内容
			if text_parts:
				finalMessage["content"] = "".join(text_parts)
			else:
				finalMessage["content"] = None  # 纯工具调用可能无文本

			# 添加 tool_calls（如果有）
			if tool_calls:
				finalMessage["tool_calls"] = tool_calls

			messages.append(finalMessage)

		elif role == "user":
			# 检查是否包含 tool_result
			has_tool_result = False
			for item in content:
				if item.get("type") == "tool_result":
					has_tool_result = True
					break

			if has_tool_result:
				# 整个消息是工具结果，应转为 role: "tool"
				for item in content:
					if item.get("type") == "tool_result":
						tool_call_id = item.get("tool_use_id")
						tool_result_content = item.get("content", [])
						is_error = item.get("is_error", False)

						# 转换 content 中的文本部分
						if isinstance(tool_result_content, list):
							result_text = "".join([t.get("text", "") for t in tool_result_content if t.get("type") == "text"])
						else:
							result_text = str(tool_result_content)

						# 构建新的 tool 消息
						tool_message = {
							"role": "tool",
							"tool_call_id": tool_call_id,
							"content": result_text
						}
						messages.append(tool_message)
			else:
				# 正常用户消息处理（text + image）
				all_text = True
				for item in content:
					if item.get("type") != "text":
						all_text = False
						break

				if all_text:
					finalMessage["content"] = "".join([item.get("text", "") for item in content])
				else:
					for item in content:
						if item.get("type") == "text":
							finalMessage["content"].append({
								"type": "text",
								"text": item.get("text", "")
							})
						elif item.get("type") == "image":
							source = item.get("source", {})
							if source.get("type") == "base64":
								media_type = source.get("media_type", "image/jpeg")
								data = source.get("data", "")
								data_url = f"data:{media_type};base64,{data}"
								finalMessage["content"].append({
									"type": "image_url",
									"image_url": {
										"url": data_url
									}
								})
				messages.append(finalMessage)

		else:
			# 其他角色（如 system）按原逻辑处理
			if isinstance(content, str):
				finalMessage["content"] = content
			else:
				all_text = True
				for item in content:
					if item.get("type") != "text":
						all_text = False
						break
				if all_text:
					finalMessage["content"] = "".join([item.get("text", "") for item in content])
				else:
					for item in content:
						if item.get("type") == "text":
							finalMessage["content"].append({
								"type": "text",
								"text": item.get("text", "")
							})
						elif item.get("type") == "image":
							source = item.get("source", {})
							if source.get("type") == "base64":
								media_type = source.get("media_type", "image/jpeg")
								data = source.get("data", "")
								data_url = f"data:{media_type};base64,{data}"
								finalMessage["content"].append({
									"type": "image_url",
									"image_url": {
										"url": data_url
									}
								})
			messages.append(finalMessage)

	data = {
		**odata, 
		"stream": True,
		"chat_id": id,
		"id": utils.request.id(),
		"model": model,
		"messages": messages,
		"features": odata.get("features", { "enable_thinking": str(odata.get("thinking", {}).get("type", "enabled")).lower() == "enabled" })
	}

	try:
		response = utils.request.chat(data, id)
	except Exception as e:
		return utils.request.response(make_response(f"上游请求失败: {e}", 502))

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

# 主入口
if __name__ == "__main__":
	log.info("---------------------------------------------------------------------")
	log.info("Z.ai 2 API https://github.com/hmjz100/Z.ai2api")
	log.info("将 Z.ai 代理为 OpenAI/Anthropic Compatible 格式")
	log.info("基于 https://github.com/kbykb/OpenAI-Compatible-API-Proxy-for-Z 重构")
	log.info("---------------------------------------------------------------------")
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