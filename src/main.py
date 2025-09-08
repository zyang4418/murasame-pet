# ==========================================
# main.py – PyQt5 桌面精灵
# 主要类：Murasame（精灵窗口）、ScreenWorker（截屏后台）、LLMWorker（推理线程）
# ==========================================

from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import QApplication, QLabel, QSystemTrayIcon, QMenu, QAction, QGraphicsOpacityEffect
from PyQt5.QtGui import QPixmap, QIcon, QImage, QFont, QPainter, QFontDatabase, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QEvent, QRect, QSize, pyqtProperty
from datetime import datetime
from src import chat, generate, utils
import hashlib
import cv2
import threading
import textwrap
import os
import time
import sys
import pyautogui

# -------------- 工具：自动换行 ------------------
def wrap_text(text, width=12):
    """
    中文按字断行，width=12 约等于 6 个汉字
    """
    return '\n'.join(textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False))

# -------------- 主窗口：ムラサメ ------------------
class Murasame(QLabel):
    def __init__(self):
        super().__init__()
        # 历史记录
        self.history = chat.identity()
        self.emotion_history = []
        self.embeddings_history = []

        # 淡入淡出用 QLabel + QGraphicsOpacityEffect
        self._fade_bg = QLabel(self)
        self._fade_fg = QLabel(self)
        for lbl in (self._fade_bg, self._fade_fg):
            lbl.setAttribute(Qt.WA_TranslucentBackground)
            lbl.setVisible(False)
            lbl.setGeometry(self.rect())
            lbl.lower()     # 放最底层
        self._fade_bg_effect = QGraphicsOpacityEffect(self._fade_bg)
        self._fade_fg_effect = QGraphicsOpacityEffect(self._fade_fg)
        self._fade_bg.setGraphicsEffect(self._fade_bg_effect)
        self._fade_fg.setGraphicsEffect(self._fade_fg_effect)
        self._fade_anim = None      # QPropertyAnimation

        # 窗口属性：无边框、置顶、透明背景
        self.setWindowFlags(Qt.FramelessWindowHint |
                            Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 初始立绘：ムラサメb 便衣+微笑+头发
        cv_img = generate.generate_fgimage(target="ムラサメb",
                                           embeddings_layers=[1717, 1475, 1261])
        pixmap = self.cvimg_to_qpixmap(cv_img)
        pixmap = pixmap.scaled(pixmap.width(
        ) // 2, pixmap.height() // 2, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pixmap)
        self.resize(pixmap.size())

        # 淡入淡出动画参数
        self._xfade_old = None
        self._xfade_new = None
        self._xfade_t = 1.0
        self._xfade_anim = None

        # 鼠标拖拽
        self.mousePressEvent = self.start_move
        self.mouseMoveEvent = self.on_move
        self.offset = None
        self.touch_head = False     # 是否正在摸头
        self.head_press_x = None

        # 文字聊天
        self.display_text = ""      # 当前已显示文字
        self.text_font = QFont()
        self.text_font.setFamily("思源黑体 CN Bold")
        self.text_font.setPointSize(24)
        self.text_x_offset = 0
        self.text_y_offset = 0
        QFontDatabase.addApplicationFont("../res/思源黑体Bold.otf")

        # 打字机计时器
        self.full_text = ""     # 完整句子
        self.typing_timer = QTimer()
        self.typing_timer.timeout.connect(self._typing_step)
        self.typing_interval = 40   # ms
        self._typing_index = 0
        self.typing_prefix = ""     # 如「丛雨：」

        # 中文输入模式
        self.setAttribute(Qt.WA_InputMethodEnabled, True)
        self.input_mode = False
        self.input_buffer = ""      # 已上屏文字
        self.preedit_text = ""      # 预编辑文字

        self.latest_response = ""   # 最后一次 AI 回复，用于失焦恢复

    # -------------- 窗口激活/失焦 ------------------
    def event(self, event):
        """
        失焦时自动退出输入模式，恢复 AI 回复显示
        激活时暂停截屏线程，避免抢焦点
        """
        if event.type() == QEvent.WindowActivate:
            print("桌宠窗口激活")
            screen_worker.should_capture = False
            if hasattr(screen_worker, "interrupt_event"):
                screen_worker.interrupt_event.set()
        elif event.type() == QEvent.WindowDeactivate:
            print("桌宠窗口失焦")
            self.input_mode = False
            self.show_text(self.latest_response, typing=True)
            screen_worker.should_capture = True
        return super().event(event)

    # -------------- OpenCV ↔ Qt ------------------
    def cvimg_to_qpixmap(self, cv_img):
        """
        BGRA numpy → QPixmap（Qt 可直接显示）
        """
        if cv_img.shape[2] == 4:
            cv_img_bgra = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGRA)
            height, width, channel = cv_img_bgra.shape
            bytes_per_line = 4 * width
            qimg = QImage(cv_img_bgra.data, width, height,
                          bytes_per_line, QImage.Format_RGBA8888)
            return QPixmap.fromImage(qimg)

    # -------------- 鼠标交互 ------------------
    def start_move(self, event):
        """
        左键：
          - 头顶区域(0~156px) → 摸头判定
          - 下方区域(>277px) → 打开输入框
        中键：拖拽移动
        """
        if event.button() == Qt.LeftButton:
            rect = self.rect()

            if event.y() < 157:
                self.touch_head = True
                self.head_press_x = event.x()
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.touch_head = False
                self.head_press_x = None
                self.setCursor(Qt.ArrowCursor)
            if event.y() > 277:
                self.input_mode = True
                self.input_buffer = ""
                self.display_text = "【ご主人】\n  ..."
                self.update()
                return

        if event.button() == Qt.MiddleButton:
            self.offset = event.pos()
            self.setCursor(Qt.SizeAllCursor)

    def on_move(self, event):
        """
        摸头滑动超过 50px → 触发“摸头”事件
        中键拖拽 → 移动窗口
        """
        if self.touch_head and self.head_press_x is not None:
            if abs(event.x() - self.head_press_x) > 50:
                self.llm_worker = LLMWorker(
                    "主人摸了摸你的头", self.history, self.emotion_history, self.embeddings_history, role="system")
                self.llm_worker.finished.connect(self.on_llm_result)
                self.llm_worker.start()
                self.touch_head = False
        if self.offset is not None and event.buttons() == Qt.MiddleButton:
            self.move(self.pos() + event.pos() - self.offset)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.offset = None
            self.setCursor(Qt.ArrowCursor)
        if event.button() == Qt.LeftButton:
            self.touch_head = False
            self.head_press_x = None
            self.setCursor(Qt.ArrowCursor)

    # -------------- 打字机效果 ------------------
    def show_text(self, text: str,
                  x_offset: int = 140,
                  y_offset: int = -100,
                  typing: bool = True,
                  typing_prefix: str = "【丛雨】\n  "):
        """
        外部统一入口：支持直接显示 or打字机
        """
        self.text_x_offset = x_offset
        self.text_y_offset = y_offset
        self.typing_prefix = typing_prefix
        if typing:
            if typing_prefix and text.startswith(typing_prefix):
                self.full_text = text[len(typing_prefix):]
                self.display_text = typing_prefix
            else:
                self.full_text = text
                self.display_text = ""
            self._typing_index = 0
            self.typing_timer.start(self.typing_interval)
        else:
            self.display_text = text
            self.full_text = text
            self.typing_timer.stop()
            self.update()

    def _typing_step(self):
        if self._typing_index < len(self.full_text):
            self.display_text = self.typing_prefix + \
                self.full_text[:self._typing_index + 1]
            self._typing_index += 1
            self.update()
        else:
            self.typing_timer.stop()

    def paintEvent(self, event):
        if self._xfade_old is not None and self._xfade_new is not None:
            w, h = self.width(), self.height()
            img_old = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
            img_old.fill(0)
            p = QPainter(img_old)
            p.setCompositionMode(QPainter.CompositionMode_Source)
            p.drawPixmap(0, 0, self._xfade_old)
            p.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            p.fillRect(img_old.rect(), QColor(
                0, 0, 0, int((1.0 - self._xfade_t) * 255)))
            p.end()
            img_new = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
            img_new.fill(0)
            p = QPainter(img_new)
            p.setCompositionMode(QPainter.CompositionMode_Source)
            p.drawPixmap(0, 0, self._xfade_new)
            p.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            p.fillRect(img_new.rect(), QColor(
                0, 0, 0, int(self._xfade_t * 255)))
            p.end()
            blended = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
            blended.fill(0)
            p = QPainter(blended)
            p.setCompositionMode(QPainter.CompositionMode_Source)
            p.drawImage(0, 0, img_old)
            p.setCompositionMode(QPainter.CompositionMode_Plus)
            p.drawImage(0, 0, img_new)
            p.end()
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawImage(0, 0, blended)
            if self.display_text:
                painter.setRenderHint(QPainter.TextAntialiasing, True)
                painter.setFont(self.text_font)
                rect = self.rect()
                text_rect = rect.adjusted(
                    self.text_x_offset,
                    self.text_y_offset,
                    self.text_x_offset,
                    -rect.height()//2 + self.text_y_offset
                )
                align_flag = Qt.AlignLeft | Qt.AlignBottom if '\n' in self.display_text else Qt.AlignHCenter | Qt.AlignBottom

                border_size = 2
                painter.setPen(QColor(44, 22, 28))
                for dx, dy in [(-border_size, 0), (border_size, 0), (0, -border_size), (0, border_size),
                               (border_size, -border_size), (border_size, border_size),
                               (-border_size, -border_size), (-border_size, border_size)]:
                    painter.drawText(text_rect.translated(
                        dx, dy), align_flag, self.display_text)

                painter.setPen(Qt.white)
                painter.drawText(text_rect, align_flag, self.display_text)
            painter.end()
            return

        super().paintEvent(event)
        if self.display_text:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.TextAntialiasing, True)
            painter.setFont(self.text_font)
            rect = self.rect()
            text_rect = rect.adjusted(
                self.text_x_offset,
                self.text_y_offset,
                self.text_x_offset,
                -rect.height()//2 + self.text_y_offset
            )
            align_flag = Qt.AlignLeft | Qt.AlignBottom if '\n' in self.display_text else Qt.AlignHCenter | Qt.AlignBottom

            border_size = 2
            painter.setPen(QColor(44, 22, 28))
            for dx, dy in [(-border_size, 0), (border_size, 0), (0, -border_size), (0, border_size),
                           (border_size, -border_size), (border_size, border_size),
                           (-border_size, -border_size), (-border_size, border_size)]:
                painter.drawText(text_rect.translated(dx, dy),
                                 align_flag, self.display_text)

            painter.setPen(Qt.white)
            painter.drawText(text_rect, align_flag, self.display_text)
            painter.end()

    def _get_fade_progress(self) -> float:
        return self._xfade_t

    def _set_fade_progress(self, value: float):
        self._xfade_t = float(value)
        self.update()

    fadeProgress = pyqtProperty(
        float, fget=_get_fade_progress, fset=_set_fade_progress)

    # -------------- 中文输入法 ------------------
    def inputMethodQuery(self, query):
        if query == Qt.ImMicroFocus:
            rect = self.rect().adjusted(
                self.text_x_offset,
                self.text_y_offset,
                self.text_x_offset,
                -self.rect().height()//2 + self.text_y_offset
            )
            pos = self.mapToGlobal(rect.bottomLeft())
            return QRect(pos, QSize(1, 30))
        return super().inputMethodQuery(query)

    def inputMethodEvent(self, event):
        if self.input_mode:
            commit = event.commitString()       # 上屏
            preedit = event.preeditString()     # 预编辑
            if commit:
                self.input_buffer += commit
            self.preedit_text = preedit
            wrapped = wrap_text(self.input_buffer + self.preedit_text)
            self.display_text = f"【ご主人】\n  「{wrapped}」"
            self.update()
        else:
            super().inputMethodEvent(event)

    # -------------- 提交用户输入 ------------------
    def handle_user_input(self):
        if hasattr(screen_worker, "interrupt_event"):
            screen_worker.interrupt_event.set()     # 打断后台截屏
        self.llm_worker = LLMWorker(
            self.input_buffer, self.history, self.emotion_history, self.embeddings_history, role="user")
        self.llm_worker.finished.connect(self.on_llm_result)
        self.llm_worker.start()

    # -------------- 接收 AI 结果 ------------------
    def on_llm_result(self, result, history, emotion_history, embeddings_history, embeddings_layers, raw_response):
        """
        线程回来：
        1. 播放语音
        2. 打字机显示
        3. 换立绘
        """
        raw_response_md5 = hashlib.md5(raw_response.encode()).hexdigest()
        QSound.play(f"./voices/{raw_response_md5}.wav")
        self.show_text(result, typing=True)
        self.latest_response = result
        self.input_buffer = ""
        self.preedit_text = ""
        self.history = history
        self.emotion_history = emotion_history
        self.embeddings_history = embeddings_history
        self.switch_image("b", embeddings_layers)

    def keyPressEvent(self, event):
        if self.input_mode:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.input_mode = False
                self.handle_user_input()
            elif event.key() == Qt.Key_Backspace:
                if self.preedit_text:
                    pass
                else:
                    self.input_buffer = self.input_buffer[:-1]
                    wrapped = wrap_text(self.input_buffer)
                    if not wrapped.strip():
                        self.display_text = "【ご主人】\n  ..."
                    else:
                        self.display_text = f"【ご主人】\n  「{wrapped}」"
                    self.update()
            else:
                char = event.text()
                if char and not self.preedit_text:
                    self.input_buffer += char
                    wrapped = wrap_text(self.input_buffer)
                    if not wrapped.strip():
                        self.display_text = "【ご主人】\n  ..."
                    else:
                        self.display_text = f"【ご主人】\n  「{wrapped}」"
                    self.update()
        else:
            super().keyPressEvent(event)

    # -------------- 立绘切换 + 淡入淡出 ------------------
    def switch_image(self, target, embeddings_layers):
        cv_img = generate.generate_fgimage(
            target=f"ムラサメ{target}", embeddings_layers=embeddings_layers)
        pixmap_new = self.cvimg_to_qpixmap(cv_img)
        pixmap_new = pixmap_new.scaled(
            pixmap_new.width() // 2, pixmap_new.height() // 2,
            Qt.KeepAspectRatio, Qt.SmoothTransformation)

        pixmap_old = self.pixmap()
        if pixmap_old is None:      # 第一次
            self.setPixmap(pixmap_new)
            self.resize(pixmap_new.size())
            self.update()
            return

        # 淡入淡出动画
        self._xfade_old = pixmap_old
        self._xfade_new = pixmap_new
        self._xfade_t = 0.0

        if self._xfade_anim:
            self._xfade_anim.stop()
        from PyQt5.QtCore import QPropertyAnimation
        self._xfade_anim = QPropertyAnimation(self, b"fadeProgress")
        self._xfade_anim.setDuration(400)
        self._xfade_anim.setStartValue(0.0)
        self._xfade_anim.setEndValue(1.0)

        def finish():
            self.setPixmap(pixmap_new)
            self.resize(pixmap_new.size())
            self._xfade_old = None
            self._xfade_new = None
            self._xfade_anim = None
            self.update()

        self._xfade_anim.finished.connect(finish)
        self._xfade_anim.start()

# -------------- 后台截屏线程 ------------------
class ScreenWorker(QThread):
    screen_result = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.history = []       # 视觉历史
        self.should_capture = False     # 开关
        self.llmworker = None
        self.interrupt_event = threading.Event()

    def run(self):
        while self.running:
            print(self.should_capture, "should_capture")
            if self.should_capture:
                self.interrupt_event.clear()
                try:
                    screenshot = pyautogui.screenshot()
                    # 先让视觉模型描述屏幕
                    sys_prompt = '''你现在要担任一个AI桌宠的视觉识别助手，我会向你提供用户此时的屏幕截图，你要识别用户此时的行为，并进行描述。我会将你的描述以system消息提供给另外一个处理语言的AI模型。'''
                    response, _ = chat.query_image(screenshot, "现在请描述用户此时的行为", [
                        {"role": "system", "content": sys_prompt}])
                    # 再让“思考助手”决定要不要告诉桌宠
                    des, self.history = chat.think_image(
                        response, self.history)
                    if des['des']:
                        print("scr worker：", des['des'])
                        self.llmworker = LLMWorker(
                            des['des'], self.history, [], [], role="system", interrupt_event=self.interrupt_event
                        )
                        self.screen_result.emit(des['des'])

                        self.llmworker.start()
                        self.llmworker.wait()
                finally:
                    pass
            time.sleep(30)      # 每 30s 截一次

    def stop(self):
        self.running = False
        if self.llmworker and self.llmworker.isRunning():
            self.interrupt_event.set()

    def on_llm_result(self, *args):
        pass

# -------------- 推理线程（防止 UI 卡死） ------------------
class LLMWorker(QThread):
    finished = pyqtSignal(str, list, list, list, list, str)

    def __init__(self, prompt, history, emotion_history, embeddings_history, role="user", interrupt_event=None):
        super().__init__()
        self.prompt = prompt
        self.history = history
        self.role = role
        self.emotion_history = emotion_history
        self.embeddings_history = embeddings_history
        self.interrupt_event = interrupt_event

    def run(self):
        try:
            t_start = time.time()
            # 给 AI 报时
            hour = datetime.now().hour
            minute = datetime.now().minute
            if 0 <= hour < 5:
                period = "凌晨"
            elif 5 <= hour < 12:
                period = "早上"
            elif 12 <= hour < 18:
                period = "下午"
            elif 18 <= hour < 24:
                period = "晚上"
            self.history.append(
                {"role": "system", "content": f"现在是{period}{hour}点{minute}分"})

            if self.interrupt_event and self.interrupt_event.is_set():
                print("LLMWorker interrupted before start")
                return

            # 1. 文本生成
            response, history = chat.query(
                prompt=self.prompt,
                history=self.history,
                role=self.role
            )

            if self.interrupt_event and self.interrupt_event.is_set():
                print("LLMWorker interrupted before start")
                return

            # 2. 中译日
            translated = chat.get_translate(response)

            if self.interrupt_event and self.interrupt_event.is_set():
                print("LLMWorker interrupted before start")
                return

            # 3. 情感分析
            emotion, emotion_history = chat.get_emotion(
                f"用户：{self.prompt}\n丛雨：{response}", self.emotion_history)

            if self.interrupt_event and self.interrupt_event.is_set():
                print("LLMWorker interrupted before start")
                return

            # 4. 语音合成（线程不阻塞）
            tts_thread = threading.Thread(
                target=chat.generate_tts, args=(translated, emotion), daemon=True)
            tts_thread.start()

            if self.interrupt_event and self.interrupt_event.is_set():
                print("LLMWorker interrupted before start")
                return

            # 5. 立绘图层
            embeddings_layers, embeddings_history = chat.get_embedings_layers(
                response, "b", self.embeddings_history)

            if self.interrupt_event and self.interrupt_event.is_set():
                print("LLMWorker interrupted before start")
                return

            # 等待语音文件落地
            raw_response_md5 = hashlib.md5(translated.encode()).hexdigest()
            voice_path = f"./voices/{raw_response_md5}.wav"

            while not os.path.exists(voice_path):
                time.sleep(0.1)

            print(len(history), "history")
            print(embeddings_layers, "b")
            print(time.time() - t_start, "sec")
            print("Emitting ============")

            result = f"「{wrap_text(response)}」"
            self.finished.emit(result, history, emotion_history,
                               embeddings_history, embeddings_layers, translated)
        finally:
            pass

# -------------- 菜单：清空历史 ------------------
def clear_history(parent):
    from PyQt5.QtWidgets import QMessageBox
    reply = QMessageBox.question(parent, "Clear History", "Are you sure you want to clear the history?",
                                 QMessageBox.Ok | QMessageBox.Cancel)
    if reply == QMessageBox.Ok:
        murasame.history = chat.identity()
        murasame.emotion_history = []
        murasame.embeddings_history = []

# -------------- 主入口 ------------------
if __name__ == "__main__":
    history = chat.identity()

    app = QApplication(sys.argv)
    murasame = Murasame()
    murasame.move(1200, 400)        # 初始位置
    murasame.show()

    # 系统托盘
    tray_icon = QSystemTrayIcon(QIcon("icon.png"), parent=app)
    tray_menu = QMenu()

    clear_action = QAction("Clear History")
    clear_action.triggered.connect(lambda: clear_history(murasame))
    exit_action = QAction("Exit")
    exit_action.triggered.connect(app.quit)

    tray_menu.addAction(clear_action)
    tray_menu.addAction(exit_action)
    tray_icon.setContextMenu(tray_menu)
    tray_icon.show()

    # 启动问候
    murasame.show_text(murasame.latest_response, typing=True)

    # 后台视觉
    if utils.get_config()['enable_vl']:
        screen_worker = ScreenWorker()

        def handle_screen_result(des_text):
            murasame.llm_worker = LLMWorker(
                des_text, murasame.history, murasame.emotion_history, murasame.embeddings_history, role="system"
            )
            murasame.llm_worker.finished.connect(murasame.on_llm_result)
            murasame.llm_worker.start()

        screen_worker.screen_result.connect(handle_screen_result)
        screen_worker.start()

    sys.exit(app.exec_())