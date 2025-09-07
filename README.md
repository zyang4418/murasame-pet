# Murasame Pet 丛雨桌宠

本项目基于 LemonQu-GIT/MurasamePet，在原有基础上优化代码，并提供尽可能详细的说明文档，方便用户使用。本项目目前仅提供 Windows 版。

## 如何部署

### 1. 安装所需包

```powershell
pip install -r ./requirements.txt
```

### 2. 安装 Ollama（本地运行模型必须）

若没有云端可用的 OpenAI 接口（Qwen3:14b, Qwen2.5vl:7b)，那么这步是必须的，即在本地运行所有模型。

在 https://ollama.com/download 下载 Ollama 并安装。

```powershell
ollama pull qwen3:14b
ollama pull qwen2.5vl:7b
```

### 3. 下载微调模型

```powershell
python ./src/download.py
```

### 4. 部署 GPT-SoVITS

https://github.com/RVC-Boss/GPT-SoVITS

运行 ./models/Murasame_SoVITS 中的两个模型。

```powershell
python api_v2.py
```

注意，`api_v2.py` 为 `GPT-SoVITS` Repository 中的文件 (https://github.com/RVC-Boss/GPT-SoVITS/blob/main/api_v2.py)。

### 5. 运行本地 API

```powershell
python ./api.py
```

### 6. 运行主程序

```powershell
python ./src/main.py
```

### 注意

若 Ollama / api.py 不在本地运行，那么需要在 `./config.json`中修改相关 endpoint 地址。

## 如何使用

点击丛雨下半部分可以输入内容，长按鼠标按住丛雨的头部并左右移动可以摸头…
