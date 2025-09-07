# ==========================================
# api.py – FastAPI 推理网关
# ==========================================

from fastapi import FastAPI, Request
from datetime import datetime
import uvicorn
import requests
import json
import torch
from ..src.utils import get_config
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

api = FastAPI()     # 创建 FastAPI 实例

# -------------- 全局配置 ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = "0"     # 指定显卡编号
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
adapter_path = "./models/Murasame"      # LoRA 或全量微调权重路径
max_seq_length = 2048
load_in_4bit = True     # 显存不够就 4bit 量化

# -------------- 工具函数 ------------------
def load_model_and_tokenizer():
    """
    一次性加载本地模型与分词器，全局复用
    返回: model, tokenizer
    """
    print(f"加载模型权重: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        adapter_path,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        load_in_4bit=load_in_4bit,
    )
    print("模型加载完成")
    return model, tokenizer

def torch_gc():
    """手动清理 GPU 显存碎片，防止长期运行 OOM"""
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

# -------------- 路由：本地模型 ------------------
@api.post("/chat")
async def create_chat(request: Request):
    """
    本地 4bit 模型推理接口
    请求体: {"prompt": str, "history": list, "max_new_tokens": int, ...}
    返回: {"response": str, "history": list, "status": 200, "time": str}
    """
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Prompt: {prompt}')
    history = json_post_list.get('history')
    history = history + [{'role': 'user', 'content': prompt}]   # 追加当前轮

    # 构造 chat_template
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    # 流式生成（TextStreamer 会逐 token 打印）
    print("<<< ", end="", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=json_post_list.get('max_new_tokens', 2048),
            temperature=json_post_list.get('temperature', 0.9),
            top_p=json_post_list.get('top_p', 0.95),
            top_k=json_post_list.get('top_k', 20),
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id
        )

    # 截取新生成部分
    reply = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": reply})

    # 打包返回
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": reply,
        "history": history,
        "status": 200,
        "time": time
    }
    print(f'[{time}] 最终回复: {reply}')
    return answer

# -------------- 路由：ollama qwen3 ------------------
@api.post("/qwen3")
async def create_qwen3_chat(request: Request):
    """
    把请求转发给本地 ollama 的 qwen3:14b 模型
    优点：省显存 / 可 CPU 跑
    """
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    role = json_post_list.get('role', 'user')
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Prompt: {prompt}')
    history = json_post_list.get('history')
    if prompt != "":
        history = history + [{'role': role, 'content': prompt}]

    # 直接调 ollama http api
    response = requests.post(
        f"{get_config()['endpoints']['ollama']}/api/chat",
        json={"model": "qwen3:14b",
              "messages": history,
              "stream": False,
              "options": {"keep_alive": -1}},  # 模型常驻显存
    )
    final_response = response.json()['message']['content']
    history = history + [{'role': 'assistant', 'content': final_response}]
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": final_response,
        "history": history,
        "status": 200,
        "time": time
    }
    print(f'[{time}] Final Response: {final_response}')
    return answer

# -------------- 路由：ollama 视觉模型 ------------------
@api.post("/qwenvl")
async def create_qwenvl_chat(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Prompt: {prompt}')
    history = json_post_list.get('history')

    # 构造含 image 的 messages
    if "image" in json_post_list:
        image_url = json_post_list.get('image')
        history = history + \
            [{'role': 'user', 'content': prompt, 'images': [image_url]}]
    else:
        history = history + [{'role': 'user', 'content': prompt}]

    response = requests.post(
        f"{get_config()['endpoints']['ollama']}/api/chat",
        json={"model": "qwen2.5vl:7b", "messages": history,
              "stream": False, "options": {"keep_alive": -1}},

    )
    final_response = response.json()['message']['content']
    history = history + [{'role': 'assistant', 'content': final_response}]
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": final_response,
        "history": history,
        "status": 200,
        "time": time
    }
    print(f'[{time}] Final Response: {final_response}')
    return answer

# -------------- 启动脚本 ------------------
if __name__ == '__main__':
    # 全局加载一次模型，所有路由复用
    model, tokenizer = load_model_and_tokenizer()
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    uvicorn.run(api, host='0.0.0.0', port=28565, workers=1)
