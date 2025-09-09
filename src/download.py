from modelscope.hub.snapshot_download import snapshot_download
import os
import json
from rich.console import Console

console = Console()

models_dir = '../models'
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

console.log("Downloading models...")

console.log("Downloading Murasame SoVITS ...")
snapshot_download(
    'LemonQu/Murasame_SoVITS', local_dir=os.path.join(models_dir, 'Murasame_SoVITS'))