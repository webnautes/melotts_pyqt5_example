import sys
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import os
import tempfile
from melo.api import TTS
import threading
import gc
import uuid

class ModelCache:
    _instance = None
    _model = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_model(self, device='cuda:0'):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = TTS(language='KR', device=device)
                    if torch.cuda.is_available():
                        self._model.eval()
        return self._model

class TTSWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, model, text: str, speaker_id: int, output_path: str, speed: float):
        super().__init__()
        self.model = model
        self.text = text.strip()
        self.speaker_id = speaker_id
        self.output_path = output_path
        self.speed = speed
        self._is_canceled = False

    def cancel(self):
        self._is_canceled = True
        
    def run(self):
        try:
            if not self._is_canceled:
                self.model.tts_to_file(
                    text=self.text,
                    speaker_id=self.speaker_id,
                    output_path=self.output_path,
                    speed=self.speed
                )
                if os.path.exists(self.output_path):
                    self.finished.emit(self.output_path)
                else:
                    self.error.emit("Failed to generate audio file")
        except Exception as e:
            if not self._is_canceled:
                self.error.emit(str(e))

class TTSPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.speaker_ids = None
        self.tts_worker = None
        self.current_audio_file = None
        
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.initUI()
        self.initTTS()
        
        self.media_player = QMediaPlayer()
        self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)

    def initTTS(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = ModelCache.get_instance().get_model()
        self.speaker_ids = self.model.hps.data.spk2id

    def initUI(self):
        self.setWindowTitle('Korean TTS Player')
        self.setGeometry(100, 100, 600, 400)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText('여기에 텍스트를 입력하세요...')
        layout.addWidget(self.text_edit)
        
        speed_layout = QHBoxLayout()
        speed_label = QLabel('속도:')
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(50)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(120)
        self.speed_value_label = QLabel('1.2')
        
        self.speed_slider.valueChanged.connect(
            lambda: self.speed_value_label.setText(f'{self.speed_slider.value() / 100.0:.1f}')
        )
        
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_value_label)
        layout.addLayout(speed_layout)
        
        play_button = QPushButton('재생')
        play_button.clicked.connect(self.play_text)
        layout.addWidget(play_button)

    def cleanup_old_file(self):
        if self.current_audio_file and os.path.exists(self.current_audio_file):
            try:
                self.media_player.stop()
                self.media_player.setMedia(QMediaContent())
                os.remove(self.current_audio_file)
            except Exception:
                pass

    def play_text(self):
        if not self.model or not self.text_edit.toPlainText():
            return
            
        # Cancel any ongoing TTS generation
        if self.tts_worker and self.tts_worker.isRunning():
            self.tts_worker.cancel()
            self.tts_worker.wait()

        # Clean up previous audio file
        self.cleanup_old_file()

        # Generate unique filename for new audio
        text = self.text_edit.toPlainText()
        speed = self.speed_slider.value() / 100.0
        filename = f'output_{uuid.uuid4().hex}.wav'
        temp_wav = os.path.join(self.temp_dir, filename)
            
        self.tts_worker = TTSWorker(
            self.model, text, self.speaker_ids['KR'], temp_wav, speed
        )
        self.tts_worker.finished.connect(self.play_audio_file)
        self.tts_worker.error.connect(self.handle_error)
        self.tts_worker.start()

    def play_audio_file(self, file_path: str):
        try:
            self.current_audio_file = file_path
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
            self.media_player.play()
        except Exception as e:
            self.handle_error(str(e))

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.media_player.setMedia(QMediaContent())

    def handle_error(self, error_msg: str):
        QMessageBox.critical(self, "Error", f"처리 중 오류가 발생했습니다: {error_msg}")

    def closeEvent(self, event):
        # Cancel ongoing TTS generation
        if self.tts_worker and self.tts_worker.isRunning():
            self.tts_worker.cancel()
            self.tts_worker.wait()

        # Stop playback and clear media
        self.media_player.stop()
        self.media_player.setMedia(QMediaContent())
        
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Clean up temporary directory
        try:
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except:
                        pass
            os.rmdir(self.temp_dir)
        except:
            pass
            
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = TTSPlayer()
    player.show()
    sys.exit(app.exec_())
