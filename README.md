***此项目目前正处于开发阶段，敬请期待。***

# Murasame Pet 丛雨桌宠

本项目基于 [LemonQu-GIT/MurasamePet](https://github.com/LemonQu-GIT/MurasamePet)，在原有基础上优化代码，并提供尽可能详细的说明文档，方便用户使用。

鉴于本地算力有限，推荐以云端模型 OpenAI API 调用的方式运行，相关配置需要放在 `config.json`。

目前的计划是：丛雨文本模型尽可能在本地运行，如果本地没有条件则使用通用文本模型。通用文本模型和视觉模型使用云端 API。丛雨语音模型在本地运行。
另外，还将实现角色切换功能，通过替换角色配置文件实现。该功能需要更多时间。

## 如何部署

### 1. 安装所需包

```powershell
pip install -r ./requirements.txt
```

### 2. 下载语音模型

```powershell
python ./src/download.py
```

### 3. 下载并部署 GPT-SoVITS

前往 [GPT-SoVITS 项目](https://github.com/RVC-Boss/GPT-SoVITS) 下载整合包，按照提示部署到电脑上。

### 4. 创建 config.json 配置文件

根据 [模板](./docs/cn/config_template.md) 中的提示创建 config.json 文件。

**注意：**
`config.json` 包含 API Key 等敏感信息，请勿分享给他人。此文件不被 Git 跟踪。

### 5. 运行语音模型

前往 GPT-SoVITS 运行 ./models/Murasame_SoVITS 中的模型。

### 6. 运行主程序

```powershell
python ./src/main.py
```

## 如何使用

点击丛雨下半部分可以输入内容，长按丛雨的脑袋并左右移动可以摸头。