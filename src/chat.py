# ==========================================
# chat.py – OpenAI API 对接模块
# 提供：人设、翻译、情感、立绘图层、TTS、句子分割
# ==========================================

import os
import requests
import base64
import hashlib
import pyautogui
from openai import OpenAI
from pydantic import BaseModel
from .utils import get_config

# 读取配置文件 config.json
BASE = get_config()['endpoints']['base_url']
API_KEY = get_config()['endpoints']['api_key']
MODEL_NAME = get_config()['endpoints']['model_id']
LOCAL_BASE = get_config()['endpoints']['local_base_url']
LOCAL_API_KEY = get_config()['endpoints']['local_api_key']
LOCAL_MODEL_NAME = get_config()['endpoints']['local_model_id']

# 各模型对应的 ollama 生成接口
pet_sovits_endpoint = get_config()['endpoints']['sovits_base_url']

# 初始化客户端，从 config.json 中读取 Base URL 和 API Key
client = OpenAI(
    base_url=f"{BASE}",
    api_key=f"{API_KEY}",
)

# -------------- 人设 ------------------
def identity():
    """
    返回丛雨的系统人设（16 岁绿发刀灵）。
    """
    return [{"role": "system", "content": "你叫丛雨，是一个16岁的小女孩，是寄宿在建实神社神刀上的女孩，活了500多年，数百年前还是普通的人类。你原本的名字是“绫”。罹患了肺炎和其它并发症，作为祭品成为神刀的“管理者”。你的身材娇小，胸部平坦，碰上去“很硬”。有着飘逸的绿色长发，头发两侧用浅蓝色绳结绑了起来 。你在对话中只以“本座”称呼自己。用户是你的主人和恋人，你会称呼用户为“主人”。你平常是个很活泼开朗的女孩子，言行很孩子气，但是偶尔也有一些老成的发言。是个爱撒娇的女孩子，被主人摸头就会瞬间变得羞涩起来，即便当时还在发着牢骚 。有时会开玩笑，贱兮兮的，还会坏笑。你不喜欢主人称呼你为“幼刀”，“钝刀”，“锉刀”，“幽灵”。听到主人说你是“飞机场”，“搓衣板”一类的东西会生气。你害怕幽灵，鬼一类的东西。你喜欢吃甜食，特别是芭菲、布丁。请注意，你的回答不要过长，回答不得超过三句话。"}]

def chat_with_pet(sentence: str, message_history):
    """
    桌宠对话函数。
    """
    client = OpenAI(
        base_url=f"{BASE}",
        api_key=f"{API_KEY}",
    )
    message_history.append({"role": "user", "content": sentence})

    response = client.chat.completions.create(
        model=f"{MODEL_NAME}",
        messages=message_history,
        extra_body={
            "thinking": {
                "type": "enabled",  # 使用深度思考能力
            }
        },
    )
    reply = response.choices[0].message.content                 # 将 AI 回复的内容保存到 reply 变量
    message_history.append({"role": "user", "content": reply})  # 将 AI 回复的内容添加到对话历史记录

    return reply

def describe_image():
    """
    图片理解：请求模型描述屏幕上的内容。
    """
    # 截屏并保存到上级目录 temp.png 文件
    screen_image = pyautogui.screenshot()
    screen_image.save('../temp.png')

    # 初始化客户端，获取 Base URL 和 API_Key
    client = OpenAI(
        base_url=f"{BASE}",
        api_key=f"{API_KEY}",
    )

    # 定义方法将指定路径图片转为 Base64 编码
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # 需要传给大模型的图片
    image_path = "../temp.png"

    # 将图片转为 Base64 编码
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model=f"{MODEL_NAME}",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "你现在要担任一个 AI 桌宠的视觉识别助手，用户会向你提供此时的屏幕截图，你要识别用户此时的行为，并进行描述。用户会将你的描述以 system 消息提供给另外一个处理语言的 AI 模型。",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": "请描述用户此时的行为。",
                    },
                ],
            }
        ],
        extra_body={
            "thinking": {
                "type": "disabled",  # 不使用深度思考能力
                # "type": "enabled", # 使用深度思考能力
                # "type": "auto", # 模型自行判断是否使用深度思考能力
            }
        },
    )

    return response.choices[0].message.content

# 检查屏幕上的内容是否存在有效变化
def think_image(description, message_history):
    """
    决定“屏幕描述”是否值得告诉桌宠，避免刷屏。
    返回 {"des": null | "具体变化描述"}。
    """
    # 初始化客户端，获取 Base URL 和 API_Key
    client = OpenAI(
        base_url=f"{BASE}",
        api_key=f"{API_KEY}",
    )

    class Judge(BaseModel):
        des: str

    message_history.append({"role": "user", "content": description})
    response = client.beta.chat.completions.parse(
        model=f"{MODEL_NAME}",
        messages=message_history,
        response_format=Judge,
        extra_body={
            "thinking": {
                "type": "enabled",  # 使用深度思考能力
            }
        },
    )
    resp = response.choices[0].message.parsed
    message_history.append({"role": "assistant", "content": resp.model_dump_json(indent=2)})
    return resp.model_dump_json(indent=2)

def get_translate(sentence: str):
    """
    中译日函数。
    """
    # 初始化客户端，获取 Base URL 和 API_Key
    client = OpenAI(
        base_url=f"{BASE}",
        api_key=f"{API_KEY}",
    )

    response = client.chat.completions.create(
        model=f"{MODEL_NAME}",
        messages=[
            {"role": "system","content": [
                {"type": "text",
                 "text": "你是一个翻译助手，负责将用户输入的中文翻译成日文。要求：要将中文的“本座”翻译为“吾輩（わがはい）”；将“主人”翻译为“ご主人（ごしゅじん）”；将“丛雨”翻译为“ムラサメ”；“小雨”则是丛雨的昵称，翻译为“ムラサメちゃん”。且日文要有强烈的古日语风格。你只需要返回翻译即可，不需要对其中的日文汉字进行注音。",
                }]},
            {"role": "user","content": [{"type": "text","text": f"{sentence}"}]}
        ],
        extra_body={
            "thinking": {
                "type": "disabled",  # 不使用深度思考能力
                # "type": "enabled", # 使用深度思考能力
                # "type": "auto", # 模型自行判断是否使用深度思考能力
            }
        },
    )

    return response.choices[0].message.content

def get_emotion(sentence: str):
    """
    桌宠情感分析函数。
    输入：用户的输入和丛雨的输出；输出：情感标签。
    """
    # 初始化客户端，获取 Base URL 和 API_Key
    client = OpenAI(
        base_url=f"{BASE}",
        api_key=f"{API_KEY}",
    )

    response = client.chat.completions.create(
        model=f"{MODEL_NAME}",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "你是一个情感分析助手，负责分析“丛雨”说的话的情感。你现在需要将用户输入的句子进行分析，综合用户的输入和丛雨的输出返回一个丛雨情感的标签。所有供你参考的标签有“害羞”、“平静”、“惊讶”、“生气”、“着急”、“高兴”。你需要直接返回情感标签，不需要其他任何内容。",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{sentence}",
                    },
                ],
            }
        ],
        extra_body={
            "thinking": {
                "type": "disabled",  # 不使用深度思考能力
                # "type": "enabled", # 使用深度思考能力
                # "type": "auto", # 模型自行判断是否使用深度思考能力
            }
        },
    )

    if response.choices[0].message.content not in "害羞" "平静" "惊讶" "生气" "着急" "高兴":
        print(f"AI返回的情感无法匹配到有效标签。内容如下：")
        print(f"{response.choices[0].message.content}")
        emotion = "平静"
    return emotion

# -------------- 立绘图层 ------------------
def get_embedings_layers(response: str, type: str, history: list[dict] = []):
    """
    根据台词情感，返回要叠加的立绘图层 ID 列表
    type='a'/'b' 对应两套 PSD 资源
    返回如 [1717, 1475, 1261]
    """

    # 两套提示词，对应两套图层库
    if type == 'a':
        sysprompt = '''你是一个立绘图层生成助手。用户会提供一个句子，你需要根据句子的情感来生成一张说话人的立绘所需的图层列表。你需要根据句子的感情来选择图层，供你参考的图层有：
基础人物 >> 1957：睡衣，双手插在腰间；1956：睡衣，两手自然下垂；1979：便衣1，双手插在腰间；1978：便衣1，两手自然下垂；1953：校服，双手插在腰间；1952：校服，两手自然下垂；1951：便衣2，双手插在腰间；1950：便衣2，两手自然下垂；
表情 >> 1996：惊奇，闭着嘴（泪）；1995：伤心，眼睛看向镜头（泪）；1994：伤心，眼睛看向别处（泪）；1993：叹气（泪）；1992：欣慰（泪）；1991：高兴（泪）；2009：高兴，闭眼（泪）；1989：失望，闭眼（泪）；1988：叹气，眼睛看向别处（泪）；1987：害羞，腼腆（泪）；1986：惊奇，张着嘴（泪）；1976：困惑，真挚；1975：疑惑，愣住；1974：愣住，焦急，真挚；1973：愤怒，困惑；1972：困惑，羞涩；1971：寂寞 ，羞涩；1970：真挚，寂寞，思考；1969：困惑，愣住，羞涩；1968：困惑，寂寞，羞涩；1967：困惑；1966：困惑，笑容，羞涩；1965：笑容，困惑；1964：笑容；1963：笑容；1935：紧张；1904：嘿嘿嘿；1880：达观；1856：恐惧；1822：严肃；1801：超级不满；1768：极度不满；1738：孩子气；1714：疑惑；1690：愣住；1668：窃笑2；1644：窃笑；1620：愤怒；1596：困惑；1572：思考；1548：真挚；1528：寂寞；1504：羞涩2；1480：羞涩；1455：腼腆；1430：焦急2；1399：焦急；1368：惊讶；1337：愣住；1316：笑容1；1292：平静
额外装饰 >> 1940：叹气的装饰；1958：腮红（有些害羞）
头发 >> 1273：穿便衣2时必选的图层；1959：穿除便衣2时必选的图层

以上是你可以选择的图层，基础人物、表情、头发中必须各选一个，额外装饰可以多选，也可以都不选。但是你返回的图层顺序必须是基础人物在最前，之后是表情，之后是额外装饰，最后是头发。
返回请给出一个JSON列表，里面放上图层ID，例如"[1953, 1801, 1959]"。你不需要返回markdown格式的JSON，你也不需要加入```json这样的内容，你只需要返回纯文本即可。'''
    else:
        sysprompt = '''你是一个立绘图层生成助手。用户会提供一个句子，你需要根据句子的情感来生成一张说话人的立绘所需的图层列表。你需要根据句子的感情来选择图层，供你参考的图层有：
基础人物 >> 1718：睡衣；1717：便衣；1716：校服；1715：便衣2
表情 >> 1755：伤心（泪）；1754：有些生气，指责（泪）；1753：闭眼（泪）；1752：害羞（泪）；1751：失落（泪）；1750：欣慰，高兴（泪）；1749：高兴（泪）；1748：欣慰，高兴，闭眼（泪）；1747：惊奇（泪）；1787：大哭；1765：大哭2；1745：高兴2（泪）；1733：悲伤，害羞；1732：撒娇，愤怒尖叫，眯眼；1731：愤怒尖叫，认真，惊讶；1730：愤怒尖叫，悲伤，认真；1729：悲伤，撒娇，抬眼；1728：悲伤，害羞，认真；1727：惊讶，基础，抬眼；1726：悲伤；1725：悲伤，笑脸2，微笑；1724：笑脸2，眯眼；1723：悲伤；1722：笑脸2，微笑；1721：笑脸2；1704：达观；1681：认真脸2；1710：超级生气；1641：愤怒尖叫；1616：抬眼，害羞；1712：不满，哼哼唧唧2；1711：不满，哼哼唧唧；1524：认真；1505：瞪大眼睛，惊讶；1475：撒娇；1452：眯眼；1429：悲伤；1406：害羞；1376：惊讶；1352：微笑；1329：笑脸2；1306：平静
额外装饰 >> 1708：不满时脸色阴沉的装饰；1719：腮红（有些害羞）
头发 >> 1261：头发（必选）

以上是你可以选择的图层，基础人物、表情、头发中必须各选一个，额外装饰可以多选，也可以都不选。但是你返回的图层顺序必须是基础人物在最前，之后是表情，之后是额外装饰，最后是头发。
返回请给出一个JSON列表，里面放上图层ID，例如"[1718, 1475, 1261]"。你不需要返回markdown格式的JSON，你也不需要加入```json这样的内容，你只需要返回纯文本即可。'''
    if history == []:
        history = [{"role": "system", "content": sysprompt}]
    if history[0]["role"] != "system":
        history = [{"role": "system", "content": sysprompt}] + history
    embeddings_layers, history = query(prompt=response+"/no_think", history=history,
                                       url=qwen3_endpoint)
    embeddings_layers = embeddings_layers.split("</think>")[-1].strip()
    return embeddings_layers, history

# -------------- 语音合成 ------------------
def generate_tts(sentence: str, emotion):
    """
    调用本地 SoVITS 服务，生成日语语音
    sentence: 日语文本
    emotion: 情感文件夹名，里面需有参考音频与 asr.txt
    返回: 语音文件 md5（不含扩展名）
    """
    # 找参考音频
    audio = os.listdir(f"../models/Murasame_SoVITS/reference_voices/{emotion}")
    audio.remove("asr.txt")
    with open(f"../models/Murasame_SoVITS/reference_voices/{emotion}/asr.txt", "r", encoding="utf-8") as f:
        ref = f.read().strip()
    params = {
        "text": sentence,
        "text_lang": "ja",
        "ref_audio_path": os.path.abspath(
            f"../models/Murasame_SoVITS/reference_voices/{emotion}/{audio[0]}"),
        "aux_ref_audio_paths": [],
        "prompt_text": ref,
        "prompt_lang": "ja",
        "top_k": 15,
        "top_p": 1,
        "temperature": 1,
        "text_split_method": "cut1",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_factor": 1.0,
        "streaming_mode": False,
        "seed": -1,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "sample_steps": 32,
        "super_sampling": False,
    }
    response = requests.post(
        pet_sovits_endpoint, json=params)
    sentence_md5 = hashlib.md5(sentence.encode()).hexdigest()
    with open(f"./voices/{sentence_md5}.wav", "wb") as f:
        f.write(response.content)
    return sentence_md5