#!/usr/bin/env python3
"""
translate_video_gui.py: Qt GUI for translate_video pipeline.
"""
import sys
import os
import tempfile
import subprocess
import json
import threading
import argparse

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QListWidget,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QProgressDialog,
)
from PyQt5.QtCore import Qt, QUrl, QProcess
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

import whisper

try:
    WHISPER_MODELS = ["turbo"] + whisper.available_models()
except Exception:
    WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
from PyQt5.QtWidgets import QComboBox  # ensure QComboBox available

# Coqui TTS model list and default
try:
    from TTS.api import TTS
    from TTS.utils.manage import ModelManager

    # list available TTS models via ModelManager
    models_file = TTS.get_models_file_path()
    manager = ModelManager(models_file=models_file, progress_bar=False, verbose=False)
    TTS_MODELS = manager.list_tts_models()
except Exception:
    TTS_MODELS = []
TTS_DEFAULT = "tts_models/multilingual/multi-dataset/xtts_v2"
# ensure default multilingual XTTS2 is first
if TTS_DEFAULT not in TTS_MODELS:
    TTS_MODELS.insert(0, TTS_DEFAULT)

# Monkey-patch TTS.utils.io.load_fsspec to force full checkpoint loads under PyTorch 2.6+
try:
    import TTS.utils.io as _tio

    _orig_load_fsspec = _tio.load_fsspec

    def _load_fsspec_full(path, *args, **kwargs):
        # ensure full unpickle (not weights_only)
        kwargs.setdefault("weights_only", False)
        return _orig_load_fsspec(path, *args, **kwargs)

    _tio.load_fsspec = _load_fsspec_full
except ImportError:
    pass
# Supported target languages
LANGUAGE_LIST = ["en", "fr", "es", "de", "it", "pt", "nl", "ru", "zh", "ja"]

# reuse core CLI functions for translation and TTS
try:
    from translate_video import translate_text, select_tts_model, generate_tts_audio
except ImportError:
    import sys, os

    sys.path.append(os.path.dirname(__file__))
    from translate_video import translate_text, select_tts_model, generate_tts_audio
# simpleaudio playback
try:
    import simpleaudio
except ImportError:
    simpleaudio = None


class MainWindow(QMainWindow):
    def __init__(self, input_url=None, target_lang=None, model=None, tts_model=None):
        super().__init__()
        # initial settings from CLI args
        self.initial_model = model
        self.initial_tts_model = tts_model
        self.setWindowTitle("Video Translator GUI")
        # Data holders
        self.video_path = None
        self.audio_path = None
        self.segments = []
        self.orig_texts = []
        self.trans_texts = []
        self.src_lang = "en"
        self.translators = {}
        self.tts = None
        # Speaker WAV reference for XTTS models
        self.speaker_wav = None
        self.tmpdir = tempfile.mkdtemp(prefix="tv_gui_")

        # UI elements
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        # Video URL input on its own row
        h_url = QHBoxLayout()
        h_url.addWidget(QLabel("Video URL/Path:"))
        self.url_edit = QLineEdit()
        if input_url:
            self.url_edit.setText(input_url)
        h_url.addWidget(self.url_edit)
        vbox.addLayout(h_url)

        # Settings row: Whisper model, target language, TTS model
        h_top = QHBoxLayout()
        # Whisper model selection
        h_top.addWidget(QLabel("Whisper Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(WHISPER_MODELS)
        # default selection: CLI arg > 'turbo' if available > first
        if self.initial_model in WHISPER_MODELS:
            self.model_combo.setCurrentText(self.initial_model)
        elif "turbo" in WHISPER_MODELS:
            self.model_combo.setCurrentText("turbo")
        else:
            self.model_combo.setCurrentIndex(0)
        h_top.addWidget(self.model_combo)

        # Target language selection
        h_top.addWidget(QLabel("Target Lang:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(LANGUAGE_LIST)
        # default French
        default_lang = target_lang if target_lang else "fr"
        if default_lang in LANGUAGE_LIST:
            self.lang_combo.setCurrentText(default_lang)
        h_top.addWidget(self.lang_combo)

        # TTS model selection
        h_top.addWidget(QLabel("TTS Model:"))
        self.tts_combo = QComboBox()
        self.tts_combo.addItems(TTS_MODELS)
        # default selection from CLI or default multilingual XTTS2
        if self.initial_tts_model in TTS_MODELS:
            self.tts_combo.setCurrentText(self.initial_tts_model)
        else:
            self.tts_combo.setCurrentText(TTS_DEFAULT)
        h_top.addWidget(self.tts_combo)
        # (Speaker WAV selection removed; always use video audio as reference)

        # Load and transcribe
        self.load_btn = QPushButton("Load & Transcribe")
        self.load_btn.clicked.connect(self.load_and_transcribe)
        vbox.addLayout(h_top)
        # Load & Transcribe button on separate line
        h_load = QHBoxLayout()
        h_load.addWidget(self.load_btn)
        vbox.addLayout(h_load)
        # Video display
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video_widget = QVideoWidget()
        self.player.setVideoOutput(self.video_widget)
        self.current_end_pos = None
        self.player.positionChanged.connect(self.on_position_changed)
        vbox.addWidget(self.video_widget)

        # Segment list and editors
        h_mid = QHBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_segment_selected)
        h_mid.addWidget(self.list_widget, 1)

        edit_box = QVBoxLayout()
        edit_box.addWidget(QLabel("Original Text:"))
        self.orig_edit = QTextEdit()
        edit_box.addWidget(self.orig_edit)
        self.update_orig_btn = QPushButton("Update Original")
        self.update_orig_btn.clicked.connect(self.update_original)
        edit_box.addWidget(self.update_orig_btn)

        # Translate button between original and translated text
        self.trans_btn = QPushButton("Translate Segment")
        self.trans_btn.clicked.connect(self.translate_segment)
        edit_box.addWidget(self.trans_btn)

        edit_box.addWidget(QLabel("Translated Text:"))
        self.trans_edit = QTextEdit()
        edit_box.addWidget(self.trans_edit)

        # Preview translated audio only
        self.preview_btn = QPushButton("Preview Audio")
        self.preview_btn.clicked.connect(self.preview_audio)
        edit_box.addWidget(self.preview_btn)

        h_mid.addLayout(edit_box, 2)
        vbox.addLayout(h_mid)

        # Export and run controls
        h_bot = QHBoxLayout()
        self.export_btn = QPushButton("Export Edits JSON")
        self.export_btn.clicked.connect(self.export_edits)
        h_bot.addWidget(self.export_btn)
        self.run_btn = QPushButton("Process Video (CLI)")
        self.run_btn.clicked.connect(self.run_cli)
        h_bot.addWidget(self.run_btn)
        vbox.addLayout(h_bot)

    def mark(self, txt):
        import datetime

        with open("video_translator.log", "a") as f:
            f.write(f"{datetime.date.today()} {txt}")

    def load_and_transcribe(self):
        url = self.url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "Input Error", "Please specify a video URL/path.")
            return

        # Disable load button and clear previous segments
        self.load_btn.setEnabled(False)
        self.orig_texts = []
        self.trans_texts = []
        self.segments = []
        self.list_widget.clear()
        # Progress dialog for worker process
        self.currentDlgText = []
        self.mark("Starting load")
        self.progress = QProgressDialog("Starting...", None, 0, 0, self)
        self.progress.setWindowTitle("Load & Transcribe")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.show()
        # Start external worker process
        worker = os.path.join(os.path.dirname(__file__), "translate_video_worker.py")
        cmd = [
            sys.executable,
            worker,
            "-i",
            url,
            "--model",
            self.model_combo.currentText(),
            "--tmpdir",
            self.tmpdir,
        ]
        self.load_proc = QProcess(self)
        self.load_proc.setProcessChannelMode(QProcess.MergedChannels)
        self.load_proc.readyReadStandardOutput.connect(self.handle_load_output)
        self.load_proc.finished.connect(self.handle_load_finished)
        self.load_proc.start(cmd[0], cmd[1:])

    def handle_load_output(self):
        # Read worker output, update list progressively
        data = bytes(self.load_proc.readAllStandardOutput()).decode("utf-8")
        for line in data.splitlines():
            if line.startswith("JSONWRITTEN:"):
                # record JSON file path
                self.segments_file = line[len("JSONWRITTEN:") :]
                continue
            if line.startswith("SEGMENT:"):
                # Format: SEGMENT:<idx>:<text>
                parts = line[len("SEGMENT:") :].split(":", 1)
                try:
                    idx = int(parts[0])
                    text = parts[1]
                except Exception:
                    continue
                # Append to orig_texts and list widget
                self.orig_texts.append(text)
                self.list_widget.addItem(f"{idx}: {text}")
            else:
                # Update progress label
                self.currentDlgText.append(str(line))
        self.progress.setLabelText("\n".join(self.currentDlgText[-8:]))

    def handle_load_finished(self, exit_code, exit_status):
        self.progress.close()
        self.load_btn.setEnabled(True)
        if exit_code != 0:
            QMessageBox.critical(self, "Error", "Load & Transcribe failed.")
            return
        # Determine segments JSON file path
        seg_file = getattr(self, "segments_file", None)
        if not seg_file:
            seg_file = os.path.join(self.tmpdir, "segments.json")
        try:
            with open(seg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read segments: {e}")
            return
        # Load segments list and detected language
        segs = data.get("segments", [])
        self.src_lang = data.get("language", self.src_lang)
        self.segments = segs
        # Initialize translation texts for each segment
        self.trans_texts = ["" for _ in segs]
        # Record audio path from worker tmpdir
        self.audio_path = os.path.join(self.tmpdir, "audio.wav")
        self.mark("Finished load")

    currentDlgText = []

    def handle_run_output(self):
        data = str(bytes(self.run_proc.readAllStandardOutput()).decode("utf-8"))
        with open("video_translator.log", "a") as f:
            f.write(data)

        if data.startswith("-- "):
            self.currentDlgText = self.currentDlgText[:-1]
            data = data[3:]

        if (
            "GenerationMixin" in data
            or "ffmpeg" in data
            or "libx264" in data
            or "Press [q] to stop " in data
            or "libavcodec" in data
            or "Processing time:" in data
            or "Real-time factor:" in data
            or "attention_mask" in data
            or "Input #" in data
            or "Stream #" in data
        ):
            return
        for line in data.splitlines():
            if "Using model" in line or "already downloaded" in line:
                continue
            if len(line.strip()) > 0:
                self.currentDlgText.append(line)
        self.progress.setLabelText("\n".join(self.currentDlgText[-20:]))

    def handle_run_finished(self, exit_code, exit_status, out_file):
        self.progress.close()
        self.run_btn.setEnabled(True)
        if exit_code != 0:
            QMessageBox.critical(self, "Error", "Processing failed.")
            return
        self.video_path = out_file
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.video_path)))
        self.player.play()
        self.mark("Finished run")
        QMessageBox.information(self, "Done", f"Output video ready: {self.video_path}")

    def update_list(self):
        self.list_widget.clear()
        for i, text in enumerate(self.orig_texts):
            self.list_widget.addItem(f"{i}: {text}")

    def on_segment_selected(self, idx):
        if idx < 0 or idx >= len(self.orig_texts):
            return
        self.orig_edit.setPlainText(self.orig_texts[idx])
        self.trans_edit.setPlainText(self.trans_texts[idx])

    def update_original(self):
        idx = self.list_widget.currentRow()
        if idx < 0:
            return
        text = self.orig_edit.toPlainText().strip()
        self.orig_texts[idx] = text
        self.list_widget.item(idx).setText(f"{idx}: {text}")

    def translate_segment(self):
        idx = self.list_widget.currentRow()
        if idx < 0:
            return
        src = self.src_lang
        tgt = self.lang_combo.currentText().strip()
        text = self.orig_texts[idx]
        # perform translation using shared CLI method
        try:
            tgt_text = translate_text(text, src, tgt)
        except Exception as e:
            QMessageBox.critical(self, "Translate Error", f"Translation failed: {e}")
            return
        self.trans_texts[idx] = tgt_text
        self.trans_edit.setPlainText(tgt_text)

    def preview_audio(self):
        idx = self.list_widget.currentRow()
        # Ensure we have segments and audio available
        if (
            not hasattr(self, "audio_path")
            or not self.audio_path
            or not os.path.exists(self.audio_path)
        ):
            QMessageBox.warning(
                self,
                "Preview Error",
                "No audio available: please Load & Transcribe first.",
            )
            return
        if idx < 0 or not self.trans_texts[idx]:
            return
        if TTS is None or simpleaudio is None:
            QMessageBox.warning(
                self, "Preview Error", "TTS or simpleaudio not available."
            )
            return
        text = self.trans_texts[idx]
        # use selected TTS model
        tts_model = self.tts_combo.currentText()
        # For XTTS voice cloning, extract the original audio segment as reference
        speaker_wav = None
        if "xtts" in tts_model.lower():
            seg = self.segments[idx]
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            speaker_wav = os.path.join(self.tmpdir, f"seg_{idx}_ref.wav")
            # cut out the segment from extracted audio
            cmd_ref = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-to",
                str(end),
                "-i",
                self.audio_path,
                "-c",
                "copy",
                speaker_wav,
            ]
            subprocess.run(
                cmd_ref,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        # init or switch TTS engine
        if self.tts is None or getattr(self.tts, "model_name", None) != tts_model:
            self.tts = TTS(tts_model, progress_bar=False, gpu=False)
        wav_file = os.path.join(self.tmpdir, f"seg_{idx}.wav")
        # generate TTS audio, using cloned reference if provided
        generate_tts_audio(
            self.tts,
            text,
            wav_file,
            speaker_wav=speaker_wav,
            language=self.lang_combo.currentText(),
            preview=False,
        )
        # Play audio
        wave_obj = simpleaudio.WaveObject.from_wave_file(wav_file)
        wave_obj.play()

    def on_position_changed(self, position):
        if self.current_end_pos is not None and position >= self.current_end_pos:
            self.player.pause()
            self.current_end_pos = None

    def export_edits(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Edits JSON", "edits.json", "JSON Files (*.json)"
        )
        if not path:
            return
        data = {"orig": {}, "tgt": {}}
        for i, txt in enumerate(self.orig_texts):
            data["orig"][str(i)] = txt
        for i, txt in enumerate(self.trans_texts):
            if txt:
                data["tgt"][str(i)] = txt
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed saving edits: {e}")

    def run_cli(self):
        url = self.url_edit.text().strip()
        tgt = self.lang_combo.currentText().strip()
        edits_file, _ = QFileDialog.getOpenFileName(
            self, "Select Edits JSON", "", "JSON Files (*.json)"
        )
        if not edits_file:
            return
        out_file, _ = QFileDialog.getSaveFileName(
            self, "Save Output Video", "output.mp4", "MP4 Files (*.mp4)"
        )
        if not out_file:
            return
        self.mark("Starting run")
        # build CLI command, skipping re-transcription if segments.json available
        worker = os.path.join(os.path.dirname(__file__), "translate_video.py")
        cmd = [
            sys.executable,
            worker,
            "-i",
            url,
            "-o",
            out_file,
            "-t",
            tgt,
            "--model",
            self.model_combo.currentText(),
            "--tts-model",
            self.tts_combo.currentText(),
            "--no-edit",
            "--edits",
            edits_file,
        ]
        # Pass precomputed segments to skip Whisper transcription
        seg_file = os.path.join(self.tmpdir, "segments.json")
        # reuse previously written segments JSON if present
        if os.path.isfile(seg_file):
            cmd += ["--segments-file", seg_file, "--src-lang", self.src_lang]
        # (Debug dialog removed)
        # Progress dialog
        self.currentDlgText = []
        self.run_btn.setEnabled(False)
        self.progress = QProgressDialog("Starting processing...", None, 0, 0, self)
        self.progress.setWindowTitle("Process Video")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.show()
        # Start external CLI process
        self.run_proc = QProcess(self)
        self.run_proc.setProcessChannelMode(QProcess.MergedChannels)
        self.run_proc.readyReadStandardOutput.connect(self.handle_run_output)
        self.run_proc.finished.connect(
            lambda code, status: self.handle_run_finished(code, status, out_file)
        )
        self.run_proc.start(cmd[0], cmd[1:])


def main():
    parser = argparse.ArgumentParser(description="Video Translator GUI")
    parser.add_argument("-i", "--input", help="Video URL/Path", default=None)
    parser.add_argument(
        "-t",
        "--target-lang",
        choices=LANGUAGE_LIST,
        default="fr",
        help="Target language code (choices: %(choices)s, default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        choices=WHISPER_MODELS,
        default=WHISPER_MODELS[0],
        help="Whisper model name (choices: %(choices)s, default: %(default)s)",
    )
    parser.add_argument(
        "--tts-model",
        dest="tts_model",
        choices=TTS_MODELS,
        default=TTS_DEFAULT,
        help="Coqui TTS model name (choices: %(choices)s, default: %(default)s)",
    )
    args = parser.parse_args()
    app = QApplication(sys.argv)
    win = MainWindow(
        input_url=args.input,
        target_lang=args.target_lang,
        model=args.model,
        tts_model=args.tts_model,
    )
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
