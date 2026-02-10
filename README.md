# flight-control-sample

无人机航拍画面分析与运镜建议的示例工程，支持图像分析、语音识别和Web前端界面。

## 功能特性

- 🎥 **图像分析**：基于VLM模型分析航拍图像，生成无人机控制建议
- 🎤 **语音识别**：支持语音指令识别，实时流式显示识别结果
- 💬 **文本输入**：支持文本指令输入
- 🌐 **Web界面**：提供友好的Web前端界面，支持语音和文本输入
- 📊 **滑动窗口分析**：支持连续帧时序分析，预测下一秒控制参数

## 目录结构

- `analyze_drone_motion.py`：主程序，支持图像分析、语音识别和Web服务
- `index.html`：Web前端界面
- `cliff-video/extract_frames.sh`：从示例视频按秒抽帧脚本
- `cliff-video/frames/`：抽取的帧图片目录
- `audio/`：示例音频文件目录
- `docs/`：API文档和示例代码
- `.env_example`：环境变量示例

## 环境准备

1. 复制并填写环境变量：

```bash
cp .env_example .env
```

2. 配置以下变量：

**ARK API配置**（必需）：
- `ARK_API_KEY`：访问密钥
- `ARK_MODEL`：模型 ID，默认 `doubao-seed-1-8-251228`
- `ARK_BASE_URL`：默认 `https://ark.cn-beijing.volces.com/api/v3`

**ASR语音识别配置**（语音功能需要）：
- `ASR_APP_ID`：ASR应用ID
- `ASR_ACCESS_TOKEN`：ASR访问令牌
- `ASR_RESOURCE_ID`：ASR资源ID，默认 `volc.seedasr.sauc.duration`
- `ASR_WS_URL`：WebSocket URL，默认 `wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async`
- `ASR_SAMPLE_RATE`：采样率，默认 `16000`
- `ASR_CHANNELS`：声道数，默认 `1`

**其他配置**：
- `WINDOW_SIZE`：滑动窗口大小，默认 `5`
- `STEP_SIZE`：滑动窗口步长，默认 `1`

## 使用方法

### 1. 图像分析

#### 抽取视频帧

```bash
cd cliff-video
./extract_frames.sh
```

输出路径：`cliff-video/frames/frame_000001.png` 等。

#### 单帧分析

```bash
python3 analyze_drone_motion.py cliff-video/frames/frame_000001.png
```

#### 多帧分析（非滑动窗口模式）

```bash
python3 analyze_drone_motion.py cliff-video/frames/frame_000001.png cliff-video/frames/frame_000002.png
```

#### 连续帧滑动窗口分析

当输入图片数量不少于 `WINDOW_SIZE` 时，脚本会按滑动窗口输出每个窗口对应的第5帧时刻建议：

```bash
python3 analyze_drone_motion.py cliff-video/frames/*.png
```

输出示例：

```text
frame_000001.png..frame_000005.png => {"t":5,"v_mps":1.2,"yaw_rate_dps":-2.0,"pitch_deg":-5,"stabilization":"..."}
frame_000002.png..frame_000006.png => {"t":6,"v_mps":1.2,"yaw_rate_dps":-2,"pitch_deg":-5,"stabilization":"..."}
```

### 2. 语音识别

#### 只返回识别文字

```bash
python3 analyze_drone_motion.py --audio audio/无人机指令音频.wav --asr-only
```

输出示例：

```text
把无人机飞得高一点，确保画面平滑。
```

#### 识别文字 + 控制建议

```bash
python3 analyze_drone_motion.py --audio audio/无人机指令音频.wav
```

输出示例：

```json
{"t":1,"v_mps":0,"yaw_rate_dps":0,"pitch_deg":0,"stabilization":"保持垂直上升速度平稳，启用高度定高模式，避免机身出现前后左右偏移"}
```

### 3. 文本输入

```bash
python3 analyze_drone_motion.py --text "把无人机飞得高一点"
```

### 4. Web前端界面

启动Web服务器：

```bash
python3 analyze_drone_motion.py --serve
```

默认端口：8000，可通过 `--port` 指定：

```bash
python3 analyze_drone_motion.py --serve --port 8080
```

访问：http://localhost:8000

**Web界面功能**：
- 📝 **文本输入**：在文本框中输入指令，点击"发送"或"文本输入"按钮
- 🎤 **语音输入**：点击"语音输入"按钮开始录音，再次点击结束录音
- 📊 **实时识别**：语音识别结果实时流式显示在红色框标记的"输入指令"区域
- 🤖 **VLM推理**：识别完成后，VLM生成的控制逻辑显示在"VLM 推理结果"区域

## API接口

### 健康检查

```bash
curl -X POST http://localhost:8000/api/health \
  -H "Content-Type: application/json"
```

### 文本输入

```bash
curl -X POST http://localhost:8000/api/text \
  -H "Content-Type: application/json" \
  -d '{"text": "把无人机飞得高一点"}'
```

### 语音识别（流式）

```bash
# 需要先录制音频并转换为base64
curl -X POST http://localhost:8000/api/voice/stream \
  -H "Content-Type: application/json" \
  -d '{"audio_base64": "...", "mime_type": "audio/webm;codecs=opus"}'
```

## 代码架构

```text
                 ┌────────────────────────────┐
                 │        运行入口 CLI         │
                 │ analyze_drone_motion.py     │
                 └──────────────┬─────────────┘
                                │
                                ▼
                 ┌────────────────────────────┐
                 │        环境加载             │
                 │ .env / 环境变量             │
                 └──────────────┬─────────────┘
                                │
                                ▼
           ┌────────────────────────────────────────┐
           │             任务分支判断               │
           │  --serve → Web服务器模式              │
           │  --audio → 语音识别模式               │
           │  --text → 文本输入模式                │
           │  图像参数 → 图像分析模式               │
           └──────────────┬─────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
┌─────────────────────┐            ┌─────────────────────────┐
│  图像分析流程        │            │  语音/文本处理流程       │
│                     │            │                         │
│ 1. 图像收集          │            │ 1. 音频/文本输入         │
│ 2. 滑动窗口分析      │            │ 2. ASR识别（如音频）     │
│ 3. VLM推理          │            │ 3. VLM推理生成控制建议   │
│ 4. 输出控制建议      │            │ 4. 返回结果             │
└─────────────────────┘            └─────────────────────────┘
```

## 技术实现

### 图像分析
- 使用VLM（视觉语言模型）分析航拍图像
- 滑动窗口时序分析，预测下一秒控制参数
- 支持缓存机制，减少重复计算

### 语音识别
- 基于WebSocket的流式语音识别
- 支持实时流式显示识别结果
- 自动音频格式转换（webm → wav → pcm）
- 统一的前后端处理逻辑

### Web前端
- 响应式设计，支持多种设备
- 实时流式识别结果显示
- 语音和文本双输入模式
- SSE（Server-Sent Events）流式数据传输

## 方案优势

- ✅ 默认无需参数即可处理全部帧，降低使用门槛
- ✅ 单帧与多帧路径清晰分离，便于扩展与维护
- ✅ 滑动窗口预测第5帧，兼顾时序一致性与实时可用性
- ✅ JSON结构化输出，便于后续自动化控制或可视化
- ✅ 缓存与上一响应复用，减少重复计算与请求成本
- ✅ 统一的前后端处理逻辑，易于维护和调试
- ✅ 实时流式识别结果展示，提升用户体验

## 依赖要求

- Python 3.7+
- ffmpeg（用于音频格式转换）
- 依赖包：
  - `websockets`：WebSocket客户端
  - `aiohttp`：异步HTTP客户端（sauc_websocket_demo.py需要）

## 注意事项

1. 确保已安装 `ffmpeg`，用于音频格式转换
2. ASR功能需要配置正确的环境变量
3. Web界面需要浏览器支持 `MediaRecorder` API（现代浏览器均支持）
4. 音频文件会自动转换为16bit/16kHz/单声道格式

## 许可证

MIT License
