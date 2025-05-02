#!/usr/bin/env python3
"""
Video Translator GUI

Uses the same translation and TTS backends as translate_video.py.
Requires:
  pip install PyQt5 youtube_dl openai-whisper ffmpeg-python transformers TTS pydub
"""
import sys
import os
import argparse
import tempfile

# CLI argument defaults and choices matching translate_video.py
try:
    import whisper
    WHISPER_MODELS = whisper.available_models()
except ImportError:
    WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
# Default Whisper model: 'turbo' if available, else first available
WHISPER_DEFAULT = "turbo" if "turbo" in WHISPER_MODELS else WHISPER_MODELS[0]

# Supported target languages (Helsinki-NLP MarianMT codes)
LANGUAGE_LIST = ["en", "fr", "es", "de", "it", "pt", "nl", "ru", "zh", "ja"]

# Available Coqui TTS models and default
try:
    from TTS.api import TTS
    from TTS.utils.manage import ModelManager
    # Use ModelManager to list available TTS models
    models_file = TTS.get_models_file_path()
    manager = ModelManager(models_file=models_file, progress_bar=False, verbose=False)
    TTS_MODELS = manager.list_tts_models()
except Exception:
    TTS_MODELS = []
TTS_DEFAULT = "tts_models/multilingual/multi-dataset/xtts_v2"
# Ensure default multilingual XTTS2 is first in the list
if TTS_DEFAULT not in TTS_MODELS:
    TTS_MODELS.insert(0, TTS_DEFAULT)

parser = argparse.ArgumentParser(description="Video Translator GUI")
parser.add_argument('-i', '--input', help="Initial video URL or file path", default=None)
parser.add_argument('-t', '--target-lang', choices=LANGUAGE_LIST, default='fr',
                    help="Target language code (choices: %(choices)s, default: %(default)s)")
parser.add_argument('--model', choices=WHISPER_MODELS, default=WHISPER_DEFAULT,
                    help="Whisper model name (choices: %(choices)s, default: %(default)s)")
parser.add_argument('--tts-model', dest='tts_model', choices=TTS_MODELS, default=TTS_DEFAULT,
                    help="Coqui TTS model name (choices: %(choices)s, default: %(default)s)")
parser.add_argument('--speaker', help="Speaker name for non-XTTS TTS models")
parser.add_argument('--speaker-wav', dest='speaker_wav',
                    help="Reference WAV file for XTTS voice cloning models")
parser.add_argument('--accent-language', dest='accent_language', choices=LANGUAGE_LIST,
                    help="Language code for TTS accent hack (overrides language)")
parser.add_argument('--preview', action='store_true',
                    help="Play each segment audio during TTS generation")
args = parser.parse_args()

# reuse core functions from translate_video.py
try:
    from translate_video import (
        translate_text, select_tts_model,
        generate_tts_audio, combine_audio_segments, merge_video_audio
    )
except ImportError:
    # ensure module path
    sys.path.append(os.path.dirname(__file__))
    from translate_video import (
        translate_text, select_tts_model,
        generate_tts_audio, combine_audio_segments, merge_video_audio
    )

import youtube_dl
import whisper

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QTableWidget, QTableWidgetItem,
    QFileDialog, QComboBox
)
from PyQt5.QtCore import QUrl, Qt, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget


class VideoTranslatorGUI(QMainWindow):
    def __init__(self, args):
        super().__init__()
        # Store CLI args
        self.args = args
        # Pre-fill URL or file path
        self.url = args.input or ""
        # Whisper model to use
        self.whisper_model = args.model
        # Source and target languages
        self.src_lang = None
        self.target_lang = args.target_lang
        # TTS settings
        self.tts_model = args.tts_model
        self.speaker = args.speaker
        self.speaker_wav = args.speaker_wav
        self.accent_language = args.accent_language
        self.preview = args.preview
        # Placeholders
        self.input_file = None
        self.segments = []
        self.audio_dir = None
        # Build UI
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Video Translator")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Input row: URL or local file
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Video URL / File:"))
        self.url_input = QLineEdit(self.url)
        row1.addWidget(self.url_input)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        row1.addWidget(browse_btn)
        layout.addLayout(row1)
        # Settings: Whisper model, target language, TTS model
        row_settings = QHBoxLayout()
        row_settings.addWidget(QLabel("Whisper Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(WHISPER_MODELS)
        self.model_combo.setCurrentText(self.whisper_model)
        row_settings.addWidget(self.model_combo)
        row_settings.addWidget(QLabel("Target Lang:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(LANGUAGE_LIST)
        self.lang_combo.setCurrentText(self.target_lang)
        row_settings.addWidget(self.lang_combo)
        row_settings.addWidget(QLabel("TTS Model:"))
        self.tts_combo = QComboBox()
        # use same TTS model list as CLI version
        self.tts_combo.addItems(TTS_MODELS)
        self.tts_combo.setCurrentText(self.tts_model)
        row_settings.addWidget(self.tts_combo)
        layout.addLayout(row_settings)

        # Action buttons
        row2 = QHBoxLayout()
        self.transcribe_btn = QPushButton("Download & Transcribe")
        self.transcribe_btn.clicked.connect(self.download_and_transcribe)
        row2.addWidget(self.transcribe_btn)

        self.translate_btn = QPushButton("Translate")
        self.translate_btn.setEnabled(False)
        self.translate_btn.clicked.connect(self.translate_segments)
        row2.addWidget(self.translate_btn)

        self.generate_audio_btn = QPushButton("Generate Audio")
        self.generate_audio_btn.setEnabled(False)
        self.generate_audio_btn.clicked.connect(self.generate_audio)
        row2.addWidget(self.generate_audio_btn)

        self.replace_audio_btn = QPushButton("Replace Audio")
        self.replace_audio_btn.setEnabled(False)
        self.replace_audio_btn.clicked.connect(self.replace_audio)
        row2.addWidget(self.replace_audio_btn)

        layout.addLayout(row2)

        # Segments table: start, end, source text, translated text
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "Start", "End", "Source Text", "Translated Text"
        ])
        self.table.itemSelectionChanged.connect(self.on_segment_selected)
        layout.addWidget(self.table)

        # Video preview area
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video_widget = QVideoWidget()
        self.mediaPlayer.setVideoOutput(self.video_widget)
        layout.addWidget(self.video_widget)

        # Status bar
        self.statusBar().showMessage("Ready")
        self.resize(900, 700)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.mov *.mkv *.flv);;All Files (*)"
        )
        if file_path:
            self.url_input.setText(file_path)

    def download_and_transcribe(self):
        self.url = self.url_input.text().strip()
        if not self.url:
            self.statusBar().showMessage("Please enter a URL or file path")
            return
        # Download if URL, else use local file
        if self.url.startswith(('http://', 'https://')):
            self.statusBar().showMessage("Downloading video...")
            ydl_opts = {'outtmpl': 'input_video.%(ext)s', 'format': 'best'}
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
                self.input_file = ydl.prepare_filename(info)
        else:
            self.input_file = self.url
        # Transcribe with Whisper
        self.statusBar().showMessage("Transcribing...")
        model_name = self.model_combo.currentText()
        model = whisper.load_model(model_name)
        result = model.transcribe(self.input_file)
        # detected source language
        self.src_lang = result.get('language', None)
        self.segments = result.get('segments', [])
        self.populate_table(self.segments)
        self.translate_btn.setEnabled(True)
        self.statusBar().showMessage(f"Transcription complete (lang: {self.src_lang})")

    def populate_table(self, segments):
        self.table.setRowCount(len(segments))
        for i, seg in enumerate(segments):
            start_item = QTableWidgetItem(f"{seg['start']:.2f}")
            start_item.setFlags(start_item.flags() & ~Qt.ItemIsEditable)
            end_item = QTableWidgetItem(f"{seg['end']:.2f}")
            end_item.setFlags(end_item.flags() & ~Qt.ItemIsEditable)
            source_item = QTableWidgetItem(seg.get('text', ''))
            translated_item = QTableWidgetItem("")
            self.table.setItem(i, 0, start_item)
            self.table.setItem(i, 1, end_item)
            self.table.setItem(i, 2, source_item)
            self.table.setItem(i, 3, translated_item)

    def translate_segments(self):
        self.statusBar().showMessage("Translating segments...")
        # refresh target language
        self.target_lang = self.lang_combo.currentText()
        # perform translation per segment
        for i in range(self.table.rowCount()):
            src = self.table.item(i, 2).text()
            try:
                tgt = translate_text(src, self.src_lang, self.target_lang)
            except Exception as e:
                tgt = f"Error: {e}"
            self.table.item(i, 3).setText(tgt)
        self.generate_audio_btn.setEnabled(True)
        self.statusBar().showMessage("Translation complete")

    def on_segment_selected(self):
        items = self.table.selectedItems()
        if not items or not self.input_file:
            return
        row = items[0].row()
        start = float(self.table.item(row, 0).text())
        end = float(self.table.item(row, 1).text())
        self.mediaPlayer.setMedia(
            QMediaContent(QUrl.fromLocalFile(os.path.abspath(self.input_file)))
        )
        self.mediaPlayer.setPosition(int(start * 1000))
        self.mediaPlayer.play()
        duration_ms = int((end - start) * 1000)
        QTimer.singleShot(duration_ms, self.mediaPlayer.pause)

    def generate_audio(self):
        self.statusBar().showMessage("Generating translated audio...")
        # select TTS model
        self.tts_model = self.tts_combo.currentText()
        # initialize TTS engine
        try:
            from TTS.api import TTS
            tts_engine = TTS(self.tts_model, progress_bar=False, gpu=False)
        except Exception as e:
            self.statusBar().showMessage(f"TTS init error: {e}")
            return
        # prepare audio output directory
        audio_dir = os.path.join(tempfile.gettempdir(), "video_translator_tts")
        os.makedirs(audio_dir, exist_ok=True)
        self.audio_dir = audio_dir
        # generate per-segment audio
        for i in range(self.table.rowCount()):
            text = self.table.item(i, 3).text()
            out_wav = os.path.join(audio_dir, f"segment_{i}.wav")
            try:
                generate_tts_audio(
                    tts_engine, text, out_wav,
                    speaker=self.speaker, speaker_wav=self.speaker_wav,
                    language=self.target_lang, preview=self.preview,
                    accent_language=self.accent_language
                )
            except Exception as e:
                self.statusBar().showMessage(f"TTS error on segment {i}: {e}")
                return
        self.replace_audio_btn.setEnabled(True)
        self.statusBar().showMessage("Audio generation complete")

    def replace_audio(self):
        self.statusBar().showMessage("Replacing audio and exporting video...")
        # combine per-segment audio into single WAV
        combined_audio = os.path.join(tempfile.gettempdir(), "video_translator_combined.wav")
        try:
            combine_audio_segments(self.segments, self.audio_dir, None, combined_audio)
        except Exception as e:
            self.statusBar().showMessage(f"Audio merge error: {e}")
            return
        # choose output video path
        output_path = getattr(self.args, 'output', None)
        if not output_path:
            output_path, _ = QFileDialog.getSaveFileName(
                self, "Save Output Video", "", "MP4 Files (*.mp4);;All Files (*)"
            )
            if not output_path:
                self.statusBar().showMessage("Export canceled")
                return
        # merge video with new audio
        try:
            merge_video_audio(self.input_file, combined_audio, output_path)
        except Exception as e:
            self.statusBar().showMessage(f"Video merge error: {e}")
            return
        self.statusBar().showMessage(f"Video saved to {output_path}")


def main():
    app = QApplication(sys.argv)
    window = VideoTranslatorGUI(args)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()