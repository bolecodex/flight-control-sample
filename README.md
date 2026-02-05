# flight-control-sample

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
