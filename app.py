#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Z.ai 2 API
将 Z.ai 代理为 OpenAI/Anthropic Compatible 格式，支持免令牌、智能处理思考链、图片上传（仅登录后）等功能
基于 https://github.com/kbykb/OpenAI-Compatible-API-Proxy-for-Z 使用 AI 辅助重构。
"""

import os, re, json, base64, urllib.parse, requests, hashlib, hmac, uuid, traceback, logging
from datetime import datetime
from flask import Flask, request, Response, jsonify, make_response
from typing import Any, Dict, List, Union, Optional

from dotenv import load_dotenv
load_dotenv()

# 配置
class cfg:
	class source:
		protocol = str(os.getenv("PROTOCOL", "https:"))
		host = str(os.getenv("BASE", "chat.z.ai"))
		token = str(os.getenv("TOKEN", "")).strip()
	class api:
		port = int(os.getenv("PORT", "8080"))
		debug = str(os.getenv("DEBUG", "false")).lower() == "true"
		debug_msg = str(os.getenv("DEBUG_MSG", "false")).lower() == "true"
		think = str(os.getenv("THINK_TAGS_MODE", "reasoning"))
		anon = str(os.getenv("ANONYMOUS_MODE", "true")).lower() == "true"
	class model:
		default = str(os.getenv("MODEL", "glm-4.6"))
		mapping = {}

	@classmethod
	def headers(cls) -> Dict[str, str]:
		return {
			"Accept": "*/*",
			"Accept-Language": "zh-CN,zh;q=0.9",
			"Cache-Control": "no-cache",
			"Connection": "keep-alive",
			"Origin": f"{cls.source.protocol}//{cls.source.host}",
			"Pragma": "no-cache",
			"Referer": f"{cls.source.protocol}//{cls.source.host}/",
			"Sec-Ch-Ua": '"Microsoft Edge";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
			"Sec-Ch-Ua-Mobile": "?0",
			"Sec-Ch-Ua-Platform": '"Windows"',
			"Sec-Fetch-Dest": "empty",
			"Sec-Fetch-Mode": "cors",
			"Sec-Fetch-Site": "same-origin",
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0",
			"X-FE-Version": "prod-fe-1.0.95",
		}


# tiktoken 预加载
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tiktoken') + os.sep
os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
assert os.path.exists(os.path.join(cache_dir, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4")) # cl100k_base.tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
# 日志
logging.basicConfig(
	level=logging.DEBUG if cfg.api.debug_msg else logging.INFO,
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
			timestamp = int(datetime.now().timestamp() * 1000)
			requestId = str(uuid.uuid4())

			user = utils.request.user()
			userToken = user.get("token")
			userId = user.get("id")

			params = {
				"timestamp": timestamp,
				"requestId": requestId,
			}
			headers = {
				**cfg.headers(),
				"Authorization": f"Bearer {userToken}",
				"Content-Type": "application/json",
				"Referer": f"{cfg.source.protocol}//{cfg.source.host}/c/{chat_id}"
			}

			if userId:
				params["user_id"] = userId

				# 最后一句话
				last_user_message = ""
				for message in data.get("messages", []):
					if message.get("role") and message.get("content"):
						content = message.get("content")
						if isinstance(content, str):
							last_user_message = content

						if isinstance(content, list):
							texts = []
							for item in content:
								if isinstance(item, dict) and item.get("type") == "text":
									texts.append(item.get("text", ""))
									break
							last_user_message = "".join(texts)

				signatures = utils.request.signature({
					"requestId": requestId,
					"timestamp": timestamp,
					"user_id": userId,
				}, last_user_message)
				headers["X-Signature"] = signatures.get("signature")
				params["signature_timestamp"] = signatures.get("timestamp")
				data["signature_prompt"] = last_user_message

			log.debug("发送请求:")
			log.debug("  headers: %s", json.dumps(headers))
			log.debug("  data: %s", json.dumps(data))

			url = f"{cfg.source.protocol}//{cfg.source.host}/api/chat/completions"
			if params:
				query_string = urllib.parse.urlencode(params)
				url = f"{url}?{query_string}"

			return requests.post(url, json=data, headers=headers, stream=True)

		@staticmethod
		def image(data_url, chat_id):
			if cfg.api.anon or not data_url.startswith("data:"):
				return None

			header, encoded = data_url.split(",", 1)
			mime_type = header.split(";")[0].split(":")[1] if ":" in header else "image/jpeg"

			image_data = base64.b64decode(encoded) # 解码数据
			filename = str(uuid.uuid4())

			log.debug("上传文件: %s", filename)

			body = {
				"file": (filename, image_data, mime_type)
			}
			headers = {
				**cfg.headers(),
				"Authorization": f"Bearer {utils.request.user().get("token")}",
				"Referer": f"{cfg.source.protocol}//{cfg.source.host}/c/{chat_id}"
			}

			response = requests.post(f"{cfg.source.protocol}//{cfg.source.host}/api/v1/files/", files=body, headers=headers)

			if response.status_code == 200:
				result = response.json()
				log.debug("上传文件: %s -> %s_%s", filename, result.get("id"), result.get("filename"))
				return f"{result.get("id")}_{result.get("filename")}"
			else:
				raise Exception(f"image upload fail: {response.text}")

		@staticmethod
		def id(prefix = "msg") -> str:
			# return f"{prefix}-{int(datetime.now().timestamp()*1e9)}"
			return f"{str(uuid.uuid4())}"

		_user_cache = {}
		@staticmethod
		def user():
			headers = {
				**cfg.headers(),
				"Content-Type": "application/json"
			}
			current_token = None if cfg.api.anon else cfg.source.token

			if current_token and current_token in utils.request._user_cache:
				cached = utils.request._user_cache[current_token]
				log.debug("用户信息[缓存]: id=%s, token=%s...", cached.get("id"), current_token[:50])
				return {"id": cached.get("id"), "token": current_token}

			if not cfg.api.anon:
				headers["Authorization"] = f"Bearer {cfg.source.token}"

			response = requests.get(f"{cfg.source.protocol}//{cfg.source.host}/api/v1/auths/", headers=headers)
			if response.status_code == 200:
				data = response.json()
				userName = data.get("name")
				userId = data.get("id")
				userToken = data.get("token") if cfg.api.anon else cfg.source.token

				if userToken and userId:
					utils.request._user_cache[userToken] = {
						"id": userId,
						"name": userName
					}

				log.debug("用户信息[实时]: name=%s, id=%s, token=%s...", userName, userId, userToken[:50] if userToken else None)
				return {"id": userId, "token": userToken}
			else:
				raise Exception(f"fetch user info fail: {response.text}")

		@staticmethod
		def signature(prarms: Dict, content: str) -> Dict:
			for param in ["timestamp", "requestId", "user_id"]:
				if param not in prarms or not prarms.get(param):
					raise ValueError(f"need prarm: {param}")

			def _hmac_sha256(key: bytes, msg: bytes):
				return hmac.new(key, msg, hashlib.sha256).hexdigest()

			# content = content.strip()
			request_time = int(prarms.get("timestamp", datetime.now().timestamp() * 1000))  # 请求时间戳（毫秒）

			# 第 1 级签名
			signature_expire = request_time // (5 * 60 * 1000)  # 5 分钟粒度
			signature_1_plaintext = str(signature_expire)
			signature_1 = _hmac_sha256(b"junjie", signature_1_plaintext.encode('utf-8'))

			# 第 2 级签名
			content = base64.b64encode(content.encode('utf-8')).decode('ascii')

			signature_prarms = str(','.join([f"{k},{prarms[k]}" for k in sorted(prarms.keys())]))
			signature_2_plaintext = f"{signature_prarms}|{content}|{str(request_time)}"
			signature_2 = _hmac_sha256(signature_1.encode('utf-8'), signature_2_plaintext.encode('utf-8'))

			# 感谢 junjie 圣开源
			log.debug("生成签名: %s", signature_2)
			log.debug("  请求时间: %s", prarms.get("timestamp"))
			log.debug("  请求标识: %s", prarms.get("requestId"))
			log.debug("  用户标识: %s", prarms.get("user_id"))
			log.debug("  最后内容: %s", content[:50])
			return {
				"signature": signature_2,
				"timestamp": request_time
			}

		_models_cache = {}
		@staticmethod
		def models() -> Dict:
			"""获取模型列表"""
			current_token = utils.request.user().get('token') if cfg.api.anon else cfg.source.token

			if utils.request._models_cache:
				return utils.request._models_cache

			def format_model_name(name: str) -> str:
				"""格式化模型名"""
				if not name:
					return ""
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

			def get_model_name(source_id: str, model_name: str) -> str:
				"""获取模型名称：优先自带，其次智能生成"""

				# 处理自带系列名的模型名称
				if source_id.startswith(("GLM", "Z")) and "." in source_id:
					return source_id

				if model_name.startswith(("GLM", "Z")) and "." in model_name:
					return model_name

				# 无法识别系列名，但名称仍为英文
				if not model_name or not ('A' <= model_name[0] <= 'Z' or 'a' <= model_name[0] <= 'z'):
					model_name = format_model_name(source_id)
					if not model_name.upper().startswith(("GLM", "Z")): model_name = model_name = "GLM-" + format_model_name(source_id)

				return model_name

			def get_model_id(source_id: str, model_name: str) -> str:
				"""获取模型 ID：优先配置，其次智能生成"""
				if hasattr(cfg.model, 'mapping') and source_id in cfg.model.mapping:
					return cfg.model.mapping[source_id]

				# 找不到配置则生成智能 ID
				smart_id = model_name.lower()
				cfg.model.mapping[source_id] = smart_id
				return smart_id

			headers = {
				**cfg.headers(),
				"Authorization": f"Bearer {current_token}",
				"Content-Type": "application/json"
			}
			response = requests.get(f"{cfg.source.protocol}//{cfg.source.host}/api/models", headers=headers)
			if response.status_code == 200:
				data = response.json()
				models = []
				for m in data.get("data", []):
					if not m.get("info", {}).get("is_active", True):
						continue
					model_id = m.get("id")
					model_name = m.get("name")
					model_info = m.get("info", {})
					model_meta = model_info.get("meta", {})
					model_logo = "data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2030%2030%22%20style%3D%22background%3A%232D2D2D%22%3E%3Cpath%20fill%3D%22%23FFFFFF%22%20d%3D%22M15.47%207.1l-1.3%201.85c-.2.29-.54.47-.9.47h-7.1V7.09c0%20.01%209.31.01%209.31.01z%22%2F%3E%3Cpath%20fill%3D%22%23FFFFFF%22%20d%3D%22M24.3%207.1L13.14%2022.91H5.7l11.16-15.81z%22%2F%3E%3Cpath%20fill%3D%22%23FFFFFF%22%20d%3D%22M14.53%2022.91l1.31-1.86c.2-.29.54-.47.9-.47h7.09v2.33h-9.3z%22%2F%3E%3C%2Fsvg%3E"

					model_meta_r = {
						"profile_image_url": model_logo,
						"capabilities": model_meta.get("capabilities"),
						"description": model_meta.get("description"),
						"hidden": model_meta.get("hidden"),
						"suggestion_prompts": [{"content": item["prompt"]} for item in (model_meta.get("suggestion_prompts") or []) if isinstance(item, dict) and "prompt" in item]
					}
					models.append({
						"id": get_model_id(model_id, get_model_name(model_id, model_name)),
						"object": "model",
						"name": get_model_name(model_id, model_name),
						"meta": model_meta_r,
						"info": {
							"meta": model_meta_r
						},
						"created": model_info.get("created_at", int(datetime.now().timestamp())),
						"owned_by": "z.ai",
						"orignal": {
							"name": model_name,
							"id": model_id,
							"info": model_info
						},
						# Special For Open WebUI
						# So, Fuck you! Private!
						"access_control": None,
					})
				result = {
					"object": "list",
					"data": models,
				}
				utils.request._models_cache = result
				return result
			else:
				raise Exception(f"fetch models info fail: {response.text}")

		@staticmethod
		def response(resp):
			resp.headers.update({
				"Access-Control-Allow-Origin": "*",
				"Access-Control-Allow-Methods": "GET, POST, OPTIONS",
				"Access-Control-Allow-Headers": "Content-Type, Authorization",
			})
			return resp

		@staticmethod
		def format(data: Dict, type: str = "OpenAI"):
			odata = {**data.copy()}
			new_messages = []
			chat_id = odata.get("chat_id")
			model = odata.get("model", cfg.model.default)

			models = utils.request.models() # 请求模型信息，以获取映射设置
			# 如果找到了映射设置
			if hasattr(cfg.model, 'mapping') and model:
				# 在映射中查找值等于当前模型的键
				for source_id, mapped_id in cfg.model.mapping.items():
					if mapped_id == model and model != source_id:
						# 找到匹配，将 model 改为源 ID（键名）
						log.debug(f"模型映射: {model} -> {source_id}")
						model = source_id
						break

			# Anthropic - system 转换 role:system
			if "system" in odata:
				systems = odata["system"]
				if isinstance(systems, str):
					content = systems.lstrip('\n')
				else:
					items = []
					for item in systems:
						if item.get("type") == "text": items.append(item.get("text", "").lstrip('\n'))
					content = "\n\n".join(items)
				new_messages.append({"role": "system", "content": content})
				del odata["system"]

			# messages 处理
			for message in odata.get("messages", []):
				role = message.get("role")
				content = message.get("content", [])
				new_message = {"role": role}

				# 如果 content 类型是文本
				if isinstance(content, str):
					new_message["content"] = content
					new_messages.append(new_message)
					continue

				# 如果 content 类型是数组
				if isinstance(content, list):
					dont_append = False
					new_content: Union[str, List[Dict[Any, Any]]] = ""
					for item in content:
						type = item.get("type")
						# 如果 消息类型 为 文本
						if type == "text":
							new_content = item.get("text")
							continue

						# 如果 消息类型 为 图片
						elif type == "image_url" or type == "image":
							media_url = ""
							# 获取 OpenAI 格式下的图片链接
							if item.get("image_url", {}).get("url"):
								media_url = item.get("image_url").get("url")
							# 获取 Anthropic 格式下的图片链接
							elif item.get("source", {}).get("data"):
								source = item.get("source")
								if source.get("type") == "base64" and source.get("data"):
									media_url = f"data:{source.get("media_type", "image/jpeg")};base64,{source.get("data")}"

							def truncate_values(obj, max_len=20):
								if isinstance(obj, dict): return {k: truncate_values(v, max_len) for k, v in obj.items()}
								elif isinstance(obj, list): return [truncate_values(x, max_len) for x in obj]
								elif isinstance(obj, str): return obj[:max_len]
								else: return obj

							if not media_url:
								if isinstance(new_content, str):
									new_content = [{
										"type": "text",
										"text": new_content
									}]
								new_content.append({
									"type": "text",
									"text": f"system: image error - Unsupported format or missing URL\norignal data:{json.dumps(truncate_values(item), ensure_ascii=False)}"
								})
								continue
							# 将以 data: 编码的图片链接上传到服务器
							try:
								uploaded_url = utils.request.image(media_url, chat_id)
								if uploaded_url: media_url = uploaded_url
							except Exception as e:
								if isinstance(new_content, str):
									new_content = [{
										"type": "text",
										"text": new_content
									}]
								new_content.append({
									"type": "text",
									"text": f"system: image upload error - {e}\norignal data:{json.dumps(truncate_values(item), ensure_ascii=False)}"
								})
								continue

							if isinstance(new_content, str):
								new_content = [{
									"type": "text",
									"text": new_content
								}]
							new_content.append({
								"type": "image_url",
								"image_url": {"url": media_url}
							})

						# Anthropic - 如果 消息类型 为 助理 使用工具
						elif type == "tool_use" and role == "assistant":
							# 如果 tool_calls 为空，初始化为空列表
							if new_message.get("tool_calls") is None:
								new_message["tool_calls"] = []

							# 直接追加到 new_msg["tool_calls"]
							new_message["tool_calls"].append({
								"id": item.get("id"),
								"type": "function",
								"function": {
									"name": item.get("name"),
									"arguments": json.dumps(item.get("input", {}) or {}, ensure_ascii=False)
								}
							})
							dont_append = True

						# Anthropic - 如果 消息类型 为 工具结果
						elif type == "tool_result":
							tool_result_content = item.get("content", [])

							# 如果 工具请求结果 类型是数组
							if isinstance(tool_result_content, list):
								# 提取所有 text 类型的内容并拼接
								_parts = []
								for _item in tool_result_content:
									if _item.get("type") == "text" and _item.get("text", ""): _parts.append(_item.get("text"))
								if _parts:
									result = "".join(_parts)
							else:
								result = tool_result_content

							new_messages.append({
								"role": "tool",
								"tool_call_id": item.get("tool_use_id"),
								"content": result
							})
							dont_append = True

						# 如果 消息类型 为 其它
						else:
							if isinstance(new_content, str):
								new_content = [{
									"type": "text",
									"text": new_content
								}]
							new_content.append(item)

					if not dont_append:
						new_message["content"] = new_content
						new_messages.append(new_message)

			result = {
				**odata,
				"model": model,
				"messages": new_messages,
				"stream": True,
				"features": {
					"enable_thinking": False, # 默认思考
					**odata.get("features", {})
				},
			}

			# Qwen 的开启思考方式
			if odata.get("enable_thinking"):
				result["features"]["enable_thinking"] = str(odata.get("enable_thinking", True))
				odata.pop("enable_thinking", None)

			# Anthropic / CherryStudio-OpenAI 的开启思考方式
			if odata.get("thinking"):
				result["features"]["enable_thinking"] = str(odata.get("thinking", {}).get("type", "enabled")).lower() == "enabled"
				odata.pop("thinking", None)

			if models:
				for _model in models.get("data", []):
					if _model.get("id") == model or _model.get("orignal", {}).get("id") == model:
						# 检查该模型是否支持 thinking 能力
						if not _model.get("orignal", {}).get("info", {}).get("meta", {}).get("capabilities", {}).get("think", False):
							del result["features"]["enable_thinking"]
							# 如果 features 为空，删除整个 features 字段
							if not result["features"]:
								del result["features"]
						break


			return result

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

				if cfg.api.think == "reasoning":
					if phase == "thinking": content = re.sub(r'\n>\s?', '\n', content)
					content = re.sub(r'\n*<summary>.*?</summary>\n*', '', content)
					content = re.sub(r"<reasoning>\n*", "", content)
					content = re.sub(r"\n*</reasoning>", "", content)
				elif cfg.api.think == "think":
					if phase == "thinking": content = re.sub(r'\n>\s?', '\n', content)
					content = re.sub(r'\n*<summary>.*?</summary>\n*', '', content)
					content = re.sub(r"<reasoning>", "<think>", content)
					content = re.sub(r"</reasoning>", "</think>", content)
				elif cfg.api.think == "strip":
					content = re.sub(r'\n*<summary>.*?</summary>\n*', '', content)
					content = re.sub(r"<reasoning>\n*", "", content)
					content = re.sub(r"</reasoning>", "", content)
				elif cfg.api.think == "details":
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
					log.warning("警告: THINK_TAGS_MODE 传入了未知的替换模式，将使用 <reasoning> 标签。")

			phaseBak = phase
			if repr(content) != repr(contentBak):
				log.debug("R 内容: %s %s", phase, repr(contentBak))
				log.debug("W 内容: %s %s", phase, repr(content))
			else:
				log.debug("R 内容: %s %s", phase, repr(contentBak))

			if phase == "thinking" and cfg.api.think == "reasoning":
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

@app.route("/v1/models", methods=["GET", "POST", "OPTIONS"])
def models():
	if request.method == "OPTIONS":
		return utils.request.response(make_response())
	try:
		data = utils.request.models()
		return utils.request.response(jsonify(data))
	except Exception as e:
		log.error(traceback.format_exc())
		return utils.request.response(jsonify({
			"error": 500,
			"message": "错误: " + str(e)
		})), 500

@app.route("/v1/chat/completions", methods=["GET", "POST", "OPTIONS"])
def OpenAI_Compatible():
	try:
		if request.method == "OPTIONS":
			return utils.request.response(make_response())

		odata = request.get_json(force=True, silent=True) or {}
		# log.debug("收到请求:")
		# log.debug("  data: %s", json.dumps(odata))
		id = utils.request.id("chat")
		stream = odata.get("stream", False)
		include_usage = odata.get("stream_options", {}).get("include_usage", True)

		data = {
			**utils.request.format(odata, "OpenAI"),
			"chat_id": id,
			"id": utils.request.id(),
		}
		model = data.get("model", cfg.model.default)
		messages = data.get("messages", [])

		# 仅当需要 usage 时才计算 prompt_tokens
		prompt_tokens: int = 0
		if include_usage:
			prompt_tokens = utils.response.count("".join(
				c if isinstance(c, str) else (c.get("text", "") if isinstance(c, dict) and c.get("type") == "text" else "")
				for m in messages
				for c in ([m["content"]] if isinstance(m.get("content"), str) else (m.get("content") or []))
			))

		response = utils.request.chat(data, id)
		if response.status_code != 200:
			return utils.request.response(jsonify({
				"error": response.status_code,
				"message": response.text or None
			})), response.status_code

		if stream:
			def generate_stream():
				completion_parts = []  # 收集 content 和 reasoning_content
				for raw_chunk in utils.response.parse(response):
					delta = utils.response.format(raw_chunk, "OpenAI")
					if not delta:
						continue

					# 累积内容（用于后续 token 计算，仅当 include_usage=True）
					if include_usage:
						if "content" in delta:
							completion_parts.append(delta["content"])
						if "reasoning_content" in delta:
							completion_parts.append(delta["reasoning_content"])

					# 构造 SSE 响应
					yield f"data: {json.dumps({
						"id": utils.request.id('chatcmpl'),
						"object": "chat.completion.chunk",
						"created": int(datetime.now().timestamp() * 1000),
						"model": model,
						"choices": [{
							"index": 0,
							"delta": delta,
							"message": delta,
							"finish_reason": None
						}]
					})}\n\n"

				# 发送 finish_reason
				yield f"data: {json.dumps({
					'id': utils.request.id('chatcmpl'),
					'object': 'chat.completion.chunk',
					'created': int(datetime.now().timestamp() * 1000),
					'model': model,
					'choices': [{
						'index': 0,
						'delta': {"role": "assistant"},
						'message': {"role": "assistant"},
						'finish_reason': "stop"
					}]
				})}\n\n"

				# 发送 usage
				if include_usage:
					completion_str = "".join(completion_parts)
					completion_tokens = utils.response.count(completion_str)
					yield f"data: {json.dumps({
						'id': utils.request.id('chatcmpl'),
						'object': 'chat.completion.chunk',
						'created': int(datetime.now().timestamp() * 1000),
						'model': model,
						'choices': [],
						'usage': {
							'prompt_tokens': prompt_tokens,
							'completion_tokens': completion_tokens,
							'total_tokens': prompt_tokens + completion_tokens
						}
					})}\n\n"

				yield "data: [DONE]\n\n"

			return Response(generate_stream(), mimetype="text/event-stream")

		else:
			# 伪 - 非流式
			content_parts = []
			reasoning_parts = []

			for raw_chunk in utils.response.parse(response):
				if raw_chunk.get("data", {}).get("done"):
					break
				delta = utils.response.format(raw_chunk)
				if not delta:
					continue
				if "content" in delta:
					content_parts.append(delta["content"])
				if "reasoning_content" in delta:
					reasoning_parts.append(delta["reasoning_content"])

			final_message = {"role": "assistant"}
			completion_str = ""
			if reasoning_parts:
				reasoning_text = "".join(reasoning_parts)
				final_message["reasoning_content"] = reasoning_text
				completion_str += reasoning_text
			if content_parts:
				content_text = "".join(content_parts)
				final_message["content"] = content_text
				completion_str += content_text

			completion_tokens = utils.response.count(completion_str)

			result = {
				"id": utils.request.id("chatcmpl"),
				"object": "chat.completion",
				"created": int(datetime.now().timestamp() * 1000),
				"model": model,
				"choices": [{
					"index": 0,
					"message": final_message,
					"finish_reason": "stop"
				}]
			}

			if include_usage:
				result["usage"] = {
					"prompt_tokens": prompt_tokens,
					"completion_tokens": completion_tokens,
					"total_tokens": prompt_tokens + completion_tokens
				}

			return utils.request.response(jsonify(result))

	except Exception as e:
		log.error(traceback.format_exc())
		return utils.request.response(jsonify({
			"error": 500,
			"message": "错误: " + str(e)
		})), 500

@app.route("/v1/messages", methods=["GET", "POST", "OPTIONS"])
def Anthropic_Compatible():
	try:
		if request.method == "OPTIONS":
			return utils.request.response(make_response())

		odata = request.get_json(force=True, silent=True) or {}
		log.debug("收到请求:")
		log.debug("  data: %s", json.dumps(odata))
		id = utils.request.id("chat")
		stream = odata.get("stream", False)

		data = {
			**utils.request.format(odata, "Anthropic"),
			"chat_id": id,
			"id": utils.request.id(),
		}
		model = data.get("model", cfg.model.default)
		messages = data.get("messages", [])

		# Anthropic 流式协议要求 message_start 中包含 input_tokens，所以必须计算
		prompt_tokens = utils.response.count("".join(
			c if isinstance(c, str) else (c.get("text", "") if isinstance(c, dict) and c.get("type") == "text" else "")
			for m in messages
			for c in ([m["content"]] if isinstance(m.get("content"), str) else (m.get("content") or []))
		))

		response = utils.request.chat(data, id)
		if response.status_code != 200:
			return utils.request.response(jsonify({
				"error": response.status_code,
				"message": response.text or None
			})), response.status_code

		if stream:
			def generate_stream():
				text_parts = []
				tool_call_parts = []
				has_tool_call = False

				# message_start
				yield "event: message_start\n"
				yield f"data: {json.dumps({
					'type': 'message_start',
					'message': {
						'id': utils.request.id(),
						'type': 'message',
						'role': 'assistant',
						'model': model,
						'stop_reason': None,
						'stop_sequence': None,
						'usage': {
							'input_tokens': prompt_tokens,
							'output_tokens': 0
						}
					}
				})}\n\n"

				yield "event: content_block_start\n"
				yield f"data: {json.dumps({
					'type': 'content_block_start',
					'index': 0,
					'content_block': {'type': 'text', 'text': ''}
				})}\n\n"

				yield "event: ping\n"
				yield f"data: {json.dumps({'type': 'ping'})}\n\n"

				# 流式解析
				for raw_chunk in utils.response.parse(response):
					log.debug("哈哈哈 %s", raw_chunk)

					if raw_chunk.get("data", {}).get("done"):
						break
					delta = utils.response.format(raw_chunk, "Anthropic")
					if not delta:
						continue

					if "tool_call" in delta:
						tool_call_parts.append(delta["tool_call"])
						tool_call_str = "".join(tool_call_parts)
						try:
							tool_json = json.loads(tool_call_str)
							# 处理 arguments -> input
							if "arguments" in tool_json:
								try:
									tool_json["input"] = json.loads(tool_json["arguments"])
								except (json.JSONDecodeError, TypeError):
									log.warning("arguments 无法解析为 JSON，保留原值: %s", tool_json["arguments"])
								del tool_json["arguments"]

							log.debug("完整！调用！: %s", tool_json)
							has_tool_call = True

							# 关闭当前 text block
							yield "event: content_block_stop\n"
							yield f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

							# 开启 tool_use block
							yield "event: content_block_start\n"
							yield f"data: {json.dumps({
								'type': 'content_block_start',
								'index': 1,
								'content_block': {
									'type': 'tool_use',
									**tool_json,
									"input": None
								}
							})}\n\n"

							# 发送 input（若存在）
							if tool_json.get("input"):
								input_json_str = json.dumps(tool_json["input"])
								chunk_size = 5  # 可根据需要调整
								for i in range(0, len(input_json_str), chunk_size):
									chunk = input_json_str[i:i + chunk_size]
									yield "event: content_block_delta\n"
									yield f"data: {json.dumps({
										'type': 'content_block_delta',
										'index': 1,
										'delta': {
											'type': 'input_json_delta',
											'partial_json': chunk
										}
									})}\n\n"

							yield "event: content_block_stop\n"
							yield f"data: {json.dumps({'type': 'content_block_stop', 'index': 1})}\n\n"
							break

						except json.JSONDecodeError:
							continue  # 等待更多数据
						except Exception as e:
							raise Exception(f"tool call parse fail: {e}")

					# 纯文本内容
					if "text" in delta:
						text_parts.append(delta["text"])
						yield "event: content_block_delta\n"
						yield f"data: {json.dumps({
							'type': 'content_block_delta',
							'index': 0,
							'delta': {'type': 'text_delta', 'text': delta['text']}
						})}\n\n"

				# 计算 completion_tokens
				completion_str = "".join(text_parts)
				completion_tokens = utils.response.count(completion_str)

				# 结束 text block（如果没有 tool_call）
				if not has_tool_call:
					yield "event: content_block_stop\n"
					yield f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

					yield "event: message_delta\n"
					yield f"data: {json.dumps({
						'type': 'message_delta',
						'delta': {
							'stop_reason': 'end_turn',
							'stop_sequence': None
						},
						'usage': {
							'output_tokens': completion_tokens
						}
					})}\n\n"
				else:
					yield "event: message_delta\n"
					yield f"data: {json.dumps({
						'type': 'message_delta',
						'delta': {
							'stop_reason': 'tool_use',
							'stop_sequence': None
						},
						'usage': {
							'output_tokens': completion_tokens
						}
					})}\n\n"

				yield "event: message_stop\n"
				yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
				# yield "data: [DONE]\n\n"

			return Response(generate_stream(), mimetype="text/event-stream")

		else:
			# 伪 - 非流式
			text_parts = []
			tool_call_parts = []

			for raw_chunk in utils.response.parse(response):
				if raw_chunk.get("data", {}).get("done"):
					break
				delta = utils.response.format(raw_chunk)
				if not delta:
					continue

				if "tool_call" in delta:
					tool_call_parts.append(delta["tool_call"])
					tool_call_str = "".join(tool_call_parts)
					try:
						tool_json = json.loads(tool_call_str)
						if "arguments" in tool_json:
							try:
								tool_json["input"] = json.loads(tool_json["arguments"])
								del tool_json["arguments"]
							except (json.JSONDecodeError, TypeError):
								log.warning("arguments 无法解析为 JSON，保留原值: %s", tool_json["arguments"])

						log.debug("完整！调用！: %s", tool_json)
						completion_tokens = utils.response.count("".join(text_parts))

						return utils.request.response(jsonify({
							"id": utils.request.id(),
							"type": "message",
							"role": "assistant",
							"model": model,
							"content": [
								{"type": "text", "text": "".join(text_parts)} if text_parts else None,
								{"type": "tool_use", **tool_json}
							],
							"usage": {
								"input_tokens": prompt_tokens,
								"output_tokens": completion_tokens
							},
							"stop_sequence": None,
							"stop_reason": "tool_use",
						}))

					except json.JSONDecodeError:
						continue
					except Exception as e:
						raise Exception(f"tool call parse fail: {e}")

				if "content" in delta:
					text_parts.append(delta["content"])
				if "reasoning_content" in delta:
					text_parts.append(delta["reasoning_content"])

			# 无 tool_call，纯文本
			completion_str = "".join(text_parts)
			completion_tokens = utils.response.count(completion_str)

			return utils.request.response(jsonify({
				"id": utils.request.id(),
				"type": "message",
				"role": "assistant",
				"model": model,
				"content": [{"type": "text", "text": completion_str}] if completion_str else [],
				"usage": {
					"input_tokens": prompt_tokens,
					"output_tokens": completion_tokens
				},
				"stop_sequence": None,
				"stop_reason": "end_turn",
			}))

	except Exception as e:
		log.error(traceback.format_exc())
		return utils.request.response(jsonify({
			"error": 500,
			"message": "错误: " + str(e)
		})), 500

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
	log.info("请稍后，正在检查网络……")
	models = utils.request.models()
	log.info("---------------------------------------------------------------------")
	log.info(f"Base           {cfg.source.protocol}//{cfg.source.host}")
	log.info("Models         /v1/models")
	log.info("OpenAI         /v1/chat/completions")
	log.info("Anthropic      /v1/messages")
	log.info("---------------------------------------------------------------------")
	log.info("服务端口：%s", cfg.api.port)
	log.info("可用模型：%s", ", ".join([item["id"] for item in models.get("data", []) if "id" in item]))
	log.info("备选模型：%s", cfg.model.default)
	log.info("思考处理：%s", cfg.api.think)
	log.info("访客模式：%s", cfg.api.anon)
	log.info("调试模式：%s", cfg.api.debug)
	log.info("调试信息：%s", cfg.api.debug_msg)
	
	if cfg.api.debug:
		app.run(host="0.0.0.0", port=cfg.api.port, threaded=True, debug=True)
	else:
		from gevent import pywsgi
		pywsgi.WSGIServer(('0.0.0.0', cfg.api.port), app).serve_forever()