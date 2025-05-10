# Video Translation Tool

A Python CLI tool to translate video speech and replace audio using Whisper, MarianMT, and Coqui-TTS.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Note: The script uses HuggingFace MarianMT which requires the SentencePiece library. The above requirements.txt now includes `sentencepiece`, but if you encounter a `MarianTokenizer requires the SentencePiece library` error, you can install it manually:

   ```bash
   pip install sentencepiece
   ```

Ensure that [ffmpeg](https://ffmpeg.org/) is installed on your system and available in your PATH.

## Usage

```bash
python translate_video.py -i INPUT -o OUTPUT -t TARGET_LANG \
  [--model WHISPER_MODEL] [--tts-model TTS_MODEL] \
  [--no-edit] [--edits EDITS_JSON] [--segments-file SEGMENTS_JSON --src-lang SRC_LANG] \
  [--preview] [--accent-language LANG] [--lip-sync --wav2lip-dir DIR --wav2lip-checkpoint CKPT]
```
**Note:** The `--lip-sync` feature is experimental, may not work reliably, and can be very slow. Use with caution.

## GUI Usage

A PyQt5-based GUI is provided for interactive video translation (`translate_video_gui.py`).

Prerequisites:
  - PyQt5 (`pip install PyQt5`)
  - simpleaudio (`pip install simpleaudio`)
  - TTS (`pip install TTS`)
  - whisper (`pip install openai-whisper`)
  - transformers and sentencepiece (`pip install transformers sentencepiece`)

Launch the GUI:
```bash
python translate_video_gui.py \
  [-i INPUT_URL_OR_PATH] \
  [-t TARGET_LANG] \
  [--model WHISPER_MODEL] \
  [--tts-model TTS_MODEL]
```

Workflow:
  1. Enter a video URL or local file path.
  2. Choose Whisper and TTS models and target language.
  3. Click **Load & Transcribe**. A background worker will download the video, extract audio, transcribe segments, and stream segment captions into the list view with a progress dialog.
  4. Select segments from the list, review and update original or translated text.
  5. Click **Translate Segment** to translate individual segments, and **Preview Audio** to hear the TTS result for the current segment.
  6. Click **Export Edits JSON** to save edits, then **Process Video (CLI)** to run the full pipeline (skips re-transcription using the saved segments) and preview the final video.

The GUI uses subprocesses for long-running tasks (transcription and full CLI processing) and displays their console output in modal progress dialogs.

## Examples

```bash
# Basic translation of a local file to French:
python translate_video.py -i myvideo.mp4 -o out.mp4 -t fr

# Translate a YouTube video using a smaller Whisper model:
python translate_video.py \
  -i https://www.youtube.com/watch?v=ABC123 \
  -o out_fr.mp4 -t fr --model small

# Use XTTS-v2 multilingual voice cloning to re-speak in English (uses original audio as reference):
python translate_video.py \
  -i myvideo.mp4 -o out_xtts.mp4 -t en \
  --tts-model tts_models/multilingual/multi-dataset/xtts_v2

# Adjust accent strength (0.0=none, 1.0=full clone):
python translate_video.py \
  -i myvideo.mp4 -o out_xtts_custom.mp4 -t en \
  --tts-model tts_models/multilingual/multi-dataset/xtts_v2 \
  --speaker-wav path/to/custom_reference.wav
```

## Future Plans

- A fully-featured Qt GUI (`translate_video_gui.py`) for interactive transcription review, translation editing, audio preview, and one-click processing with progress dialogs.
- Support for accent strength adjustments.
- Lip-sync generated audio with video face detection (experimental & slow; needs optimization).