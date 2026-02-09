﻿﻿﻿# flight-control-sample

无人机航拍画面分析与运镜建议的示例工程，包含帧抽取脚本与基于 Response API 的图像分析脚本。

## 目录结构

- analyze_drone_motion.py：调用模型分析图像与连续帧序列
- cliff-video/extract_frames.sh：从示例视频按秒抽帧
- cliff-video/frames：抽取的帧图片目录
- promt.md：多帧时序分析提示词文本
- .env_example：环境变量示例

## 环境准备

1. 复制并填写环境变量：

```bash
cp .env_example .env
```

2. 配置以下变量：

- ARK_API_KEY：访问密钥
- ARK_MODEL：模型 ID，默认 doubao-seed-1-8-251228
- ARK_BASE_URL：默认 https://ark.cn-beijing.volces.com/api/v3
- WINDOW_SIZE：滑动窗口大小，默认 5
- STEP_SIZE：滑动窗口步长，默认 1

## 抽取视频帧

```bash
cd cliff-video
./extract_frames.sh
```

输出路径：cliff-video/frames/frame_000001.png 等。

## 单帧或多帧分析

单帧：

```bash
python analyze_drone_motion.py cliff-video/frames/frame_000001.png
```

多帧（非滑动窗口模式）：

```bash
python analyze_drone_motion.py cliff-video/frames/frame_000001.png cliff-video/frames/frame_000002.png
```

## 连续帧滑动窗口分析

当输入图片数量不少于 WINDOW_SIZE 时，脚本会按滑动窗口输出每个窗口对应的第 5 帧时刻建议：

```bash
python analyze_drone_motion.py cliff-video/frames/frame_000001.png cliff-video/frames/frame_000002.png cliff-video/frames/frame_000003.png cliff-video/frames/frame_000004.png cliff-video/frames/frame_000005.png
```

输出示例：

```text
frame_000001.png..frame_000005.png => {"t":4,"v_mps":1.2,"yaw_rate_dps":-0.8,"pitch_deg":-2,"stabilization":"..."}
```

## 代码架构图

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
                 ┌────────────────────────────┐
                 │        图像收集             │
                 │ 参数传入 / 默认帧目录扫描   │
                 └──────────────┬─────────────┘
                                │
                                ▼
           ┌────────────────────────────────────────┐
           │             任务分支判断               │
           │  单帧/少帧 → 单次分析                  │
           │  多帧>=WINDOW_SIZE → 滑动窗口分析       │
           └──────────────┬─────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
┌─────────────────────┐            ┌─────────────────────────┐
│ 单次分析请求构建    │            │ 滑动窗口请求构建         │
│ prompt + 图像内容   │            │ prompt + 5帧时序         │
└─────────┬───────────┘            └─────────┬───────────────┘
          │                                  │
          ▼                                  ▼
┌─────────────────────┐            ┌─────────────────────────┐
│  Response API 调用   │            │  Response API 调用       │
│ JSON 请求/响应解析   │            │ 缓存与上一响应复用        │
└─────────┬───────────┘            └─────────┬───────────────┘
          │                                  │
          ▼                                  ▼
┌─────────────────────┐            ┌─────────────────────────┐
│ 输出JSON建议         │            │ 输出每窗口第5帧建议       │
└─────────────────────┘            └─────────────────────────┘
```

## 方案优势

- 默认无需参数即可处理全部帧，降低使用门槛
- 单帧与多帧路径清晰分离，便于扩展与维护
- 滑动窗口预测第5帧，兼顾时序一致性与实时可用性
- JSON结构化输出，便于后续自动化控制或可视化
- 缓存与上一响应复用，减少重复计算与请求成本
