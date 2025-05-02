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
  [--model WHISPER_MODEL] [--tts-model TTS_MODEL] [--speaker SPEAKER] [--speaker-wav SPEAKER_WAV] \
  [--accent-strength FLOAT] [--accent-language LANG] [--no-edit]
```

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

- Qt GUI for manual correction and playback.
- Support for accent strength adjustments.
- Lip-sync generated audio with video face detection.