> [!IMPORTANT]
> 您正在查看的是 Canary 分支，此分支包含众多 Bug，建议谨慎使用。

<div align=center>
<img width="100" src="https://wsrv.nl/?url=https%3a%2f%2fz-cdn.chatglm.cn%2fz-ai%2fstatic%2flogo.svg&w=300&output=webp" />
<h1>Z.ai2api</h1>
<p>将 Z.ai 代理为 OpenAI/Anthropic Compatible 格式，支持免令牌、智能处理思考链、图片上传（登录后）等功能</p>
</div>

## 功能
- OpenAI/Anthropic Compatible 接口
  - 智能识别思考链，完美转换为多种格式
  - 智能模型标识映射（`glm-4.6` -> `GLM-4-6-API-V1`）
  - （登录后）支持上传图片，使用 GLM 识图系列模型
- Anthropic Compatible 接口
  - 智能识别工具块，转换为暂不完美的工具调用
  - 工具调用
- Models 接口
  - 支持根据官网 /api/models 生成模型列表
  - 智能选择或生成合适的模型信息返回，示例：
    | 原始 | 结果 |
    |------|------|
    | id: `GLM-4-6-API-V1`<br>name: `GLM-4.6` | id: `glm-4.6`<br>name: `GLM-4.6` |
    | id: `deep-research`<br>name: `Z1-Rumination` | id: `z1-rumination`<br>name: `Z1-Rumination` |
    | id: `glm-4-flash`<br>name: `任务专用` | id: `glm-4-flash`<br>name: `GLM-4-Flash` |
    | id: `0808-360B-DR`<br>name: `0808-360B-DR` | id: `glm-0808-360b-dr`<br>name: `GLM-0808-360b-Dr` |
  - 特别适配 Open WebUI（下述内容为默认设置，后续可在 OWB 中更改）
    - 模型默认设为公开
    - 模型 meta profile_image_url 设为 Z.ai 的 data: Logo
    - 模型根据官网 hidden 设置 hidden 属性
    - 模型根据官网 suggestion_prompts 添加 suggestion_prompts

## 要求
![Python 3.12+](https://img.shields.io/badge/3.12%2B-blue?style=for-the-badge&logo=python&label=python)
![.env](https://img.shields.io/badge/.env-%23555?style=for-the-badge&logo=.env)

## 环境
使用 `.env` 文件进行配置。

### `PROTOCOL`
  - 上游 API 基础协议
  - 默认值：`https`

### `BASE`
  - 上游 API 基础域名
  - 默认值：`chat.z.ai`

### `TOKEN`
  - 提供给上游 API 的访问令牌
  - 如果启用了 `ANONYMOUS_MODE` 可不填

### `PORT`
  - 服务对外端口
  - 默认值：`8080`

### `THINK_TAGS_MODE`
  - 思考链格式化模式
  - 默认值：`reasoning`
  - 可选 `reasoning` `think` `strip` `details`，效果如下
    - "reasoning"
      - reasoning_content: `嗯，用户……`
      - content: `你好！`
    - "think"
      - content: `<think>\n\n嗯，用户……\n\n</think>\n\n你好！`
    - "strip"
      - content: `> 嗯，用户……\n\n你好！`
    - "details"
      - content: `<details type="reasoning" open><div>\n\n嗯，用户……\n\n</div><summary>Thought for 1 seconds</summary></details>\n\n你好！`

### `ANONYMOUS_MODE`
  - 访客模式，启用后将获取随机令牌
  - 默认值：`true`
  - 访客模式下不支持上传文件调用视觉模型

### `MODEL`
  - 备选模型，在未传入模型时调用
  - 默认值：`GLM-4.5`

### `DEBUG`
  - 启用调试模式，启用后将使用 Flash 自带的开发服务器运行，否则将使用 pywsgi 运行
  - 默认值：`false`

### `DEBUG_MSG`
  - 显示调试信息，启用后将显示调试信息
  - 默认值：`false`

## 使用
```
git clone https://github.com/hmjz100/Z.ai2api.git
cd Z.ai2api
pip install -r requirements.txt
python app.py
```

## 说明
初始版本基于 https://github.com/kbykb/OpenAI-Compatible-API-Proxy-for-Z 使用 AI 辅助重构
