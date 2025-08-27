import json
from rich.console import Console
from datetime import datetime
from sys import _getframe
from os import path

console = Console()


def log(event: str, type: str, show: bool = True, save: bool = True):
    back_frame = _getframe().f_back
    if back_frame is not None:
        back_filename = path.basename(back_frame.f_code.co_filename)
        back_funcname = back_frame.f_code.co_name
        back_lineno = back_frame.f_lineno
    else:
        back_filename = "Unknown"
        back_funcname = "Unknown"
        back_lineno = "Unknown"
    now = datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    logger = f"[{time}] <{back_filename}:{back_lineno}> <{back_funcname}()> {type}: {event}"
    if type.lower() == "info":
        style = "green"
    elif type.lower() == "error":
        style = "red"
    elif type.lower() == "warning":
        style = "yellow"
    elif type.lower() == "critical":
        style = "bold red"
    elif type.lower() == "event":
        style = "#ffab70"
    else:
        style = ""

    if show:
        console.print(logger, style=style)
    if save:
        with open('latest.log', 'a', encoding='utf-8') as f:
            f.write(f'{logger}\n')


def get_config() -> dict:
    with open("./config.json", "r", encoding="utf-8") as f:
        return json.load(f)
