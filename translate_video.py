#!/usr/bin/env python3
"""
translate_video.py: Translate video subtitles and audio using Whisper, MarianMT, and Coqui-TTS.

Usage:
  python translate_video.py -i INPUT -o OUTPUT -t TARGET_LANG [--model WHISPER_MODEL]
                           [--accent-strength FLOAT] [--accent-language LANG] [--no-edit]
"""
import argparse
import os
import sys
# Available Whisper models (local openai-whisper)
try:
    import whisper
    WHISPER_MODELS = whisper.available_models()
except ImportError:
    WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large']
# Available Coqui TTS models
try:
    from TTS.api import TTS
    from TTS.utils.manage import ModelManager
    # List available TTS models via ModelManager
    models_file = TTS.get_models_file_path()
    manager = ModelManager(models_file=models_file, progress_bar=False, verbose=False)
    TTS_MODELS = manager.list_tts_models()
except Exception:
    TTS_MODELS = []
TTS_DEFAULT = 'tts_models/multilingual/multi-dataset/xtts_v2'
# Ensure default multilingual XTTS2 is first choice
if TTS_DEFAULT not in TTS_MODELS:
    TTS_MODELS.insert(0, TTS_DEFAULT)
# Supported target languages (Helsinki-NLP MarianMT)
LANGUAGE_LIST = ['en', 'fr', 'es', 'de', 'it', 'pt', 'nl', 'ru', 'zh', 'ja']

# Auto-activate .venv if present in script directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_venv_dir = os.path.join(_script_dir, '.venv')
_venv_py = os.path.join(_venv_dir, 'bin', 'python3')
if os.path.isdir(_venv_dir) and sys.executable != _venv_py:
    if os.path.isfile(_venv_py) and os.access(_venv_py, os.X_OK):
        os.execv(_venv_py, [_venv_py] + sys.argv)
    print(f"Error: .venv python not found or not executable at {_venv_py}", file=sys.stderr)
    sys.exit(1)

import tempfile
import subprocess
import shutil
import json

from pathlib import Path

def download_video(url, download_dir):
    """
    Download video from URL (YouTube, TikTok) using yt-dlp into download_dir.
    Returns path to downloaded video file.
    """
    import subprocess

    # Ensure yt-dlp is installed
    cmd = ["yt-dlp", "--version"]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("Error: yt-dlp is required to download videos. Please install yt-dlp.", file=sys.stderr)
        sys.exit(1)
    # Download best quality
    out_template = str(Path(download_dir) / "video.%(ext)s")
    cmd = [
        "yt-dlp", url,
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "-o", out_template
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Error: failed to download video from {url}", file=sys.stderr)
        sys.exit(1)
    # Find downloaded file
    for f in os.listdir(download_dir):
        if f.startswith("video."):
            return os.path.join(download_dir, f)
    print("Error: downloaded video not found.", file=sys.stderr)
    sys.exit(1)

def extract_audio(video_path, audio_path):
    """
    Extract audio from video using ffmpeg, one channel, 16kHz WAV.
    """
    import subprocess

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000", audio_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"Error: failed to extract audio from {video_path}", file=sys.stderr)
        sys.exit(1)

def transcribe_audio(audio_path, model_name):
    """
    Transcribe audio using Whisper model. Returns dict with 'segments' list and 'language'.
    """
    try:
        import whisper
    except ImportError:
        print("Error: openai-whisper is required. Please install via 'pip install openai-whisper'.", file=sys.stderr)
        sys.exit(1)
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result

def translate_text(text, src_lang, tgt_lang):
    """
    Translate text from src_lang to tgt_lang using MarianMT.
    """
    from transformers import MarianMTModel, MarianTokenizer

    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading translation model {model_name}: {e}", file=sys.stderr)
        sys.exit(1)
    batch = tokenizer([text], return_tensors="pt", padding=True)
    gen = model.generate(**batch)
    tgt_text = tokenizer.decode(gen[0], skip_special_tokens=True)
    return tgt_text

def select_tts_model(target_lang):
    """
    Select a Coqui TTS model for the target language from available models.
    """
    try:
        from TTS.api import TTS
        from TTS.utils.manage import ModelManager
    except ImportError:
        print("Error: Coqui TTS (TTS) is required. Please install via 'pip install TTS'.", file=sys.stderr)
        sys.exit(1)

    # Load available TTS model names
    models_file = TTS.get_models_file_path()
    manager = ModelManager(models_file=models_file, progress_bar=False, verbose=False)
    models = manager.list_tts_models()
    # 1) try language-specific models
    lang_matches = [m for m in models if f"/{target_lang}/" in m]
    if lang_matches:
        return lang_matches[0]
    # 2) fallback to any multilingual model
    multi_matches = [m for m in models if m.startswith("tts_models/multilingual/")]
    if multi_matches:
        print(f"Warning: no language-specific TTS model for '{target_lang}', using multilingual model '{multi_matches[0]}'", file=sys.stderr)
        return multi_matches[0]
    # 3) no model found
    print(f"Error: no TTS model found for language '{target_lang}'", file=sys.stderr)
    sys.exit(1)

def generate_tts_audio(tts, text, output_path, speaker=None, speaker_wav=None,
                       language=None, preview=False, accent_language=None):
    """
    Generate TTS audio for a segment.
    If the model is a multilingual XTTS clone, uses voice cloning via speaker_wav
    and applies accent_language (if provided) to select phoneme rules.
    Otherwise, performs standard TTS synthesis with optional accent_language on
    multilingual models.
    """
    import os, subprocess, shutil

    # Simplified TTS invocation: XTTS clone or plain TTS with optional accent-language
    model_name = getattr(tts, 'model_name', '') or ''
    # XTTS clone path
    if 'xtts' in model_name.lower():
        if not speaker_wav:
            import sys
            print(f"Error: a reference WAV (--speaker-wav) is required for XTTS model '{model_name}'", file=sys.stderr)
            sys.exit(1)
        tts_kwargs = {'text': text, 'speaker_wav': speaker_wav, 'file_path': output_path}
        if getattr(tts, 'is_multi_lingual', False):
            lang = accent_language or language
            if lang:
                tts_kwargs['language'] = lang
        tts.tts_to_file(**tts_kwargs)
        if preview:
            cmd = ['afplay', output_path] if shutil.which('afplay') else (['ffplay', '-autoexit', '-nodisp', output_path] if shutil.which('ffplay') else None)
            if cmd:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    # Non-XTTS fallback
    tts_kwargs = {'text': text, 'file_path': output_path}
    if speaker_wav:
        tts_kwargs['speaker_wav'] = speaker_wav
    elif speaker:
        tts_kwargs['speaker'] = speaker
    if accent_language and getattr(tts, 'is_multi_lingual', False):
        tts_kwargs['language'] = accent_language
    elif language and getattr(tts, 'is_multi_lingual', False):
        tts_kwargs['language'] = language
    tts.tts_to_file(**tts_kwargs)
    if preview:
        cmd = ['afplay', output_path] if shutil.which('afplay') else (['ffplay', '-autoexit', '-nodisp', output_path] if shutil.which('ffplay') else None)
        if cmd:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def combine_audio_segments(segments, audio_dir, total_duration, output_audio):
    """
    Combine per-segment WAV files into a single continuous audio by concatenation.
    """
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    for i in range(len(segments)):
        wav_path = os.path.join(audio_dir, f"segment_{i}.wav")
        if not os.path.exists(wav_path):
            continue
        seg_audio = AudioSegment.from_wav(wav_path)
        combined += seg_audio
    # Export combined audio
    combined.export(output_audio, format="wav")

def _get_media_duration(path):
    """
    Return duration in seconds of a media file using ffprobe.
    """
    import subprocess
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ], stderr=subprocess.DEVNULL)
        return float(out.strip())
    except Exception:
        return None

def merge_video_audio(video_path, audio_path, output_path):
    """
    Merge video with new audio, adjusting video speed to match audio duration.
    """
    import subprocess

    # Determine durations
    vid_dur = _get_media_duration(video_path)
    aud_dur = _get_media_duration(audio_path)
    # Compute speed ratio: new_pts = old_pts * (aud_dur / vid_dur)
    ratio = None
    if vid_dur and aud_dur and vid_dur > 0:
        ratio = aud_dur / vid_dur

    # Build ffmpeg command
    if ratio is None or abs(ratio - 1.0) < 0.01:
        # durations match (or no valid ratio): simple merge
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_path
        ]
    else:
        # adjust video speed: stretch video to match new audio duration
        # setpts multiplies frame timestamps
        filter_spec = f"[0:v]setpts={ratio}*PTS[v]"
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex", filter_spec,
            "-map", "[v]", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy", "-shortest", output_path
        ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Error: failed to merge and speed-adjust video into {output_path}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Translate video speech and replace audio.")
    parser.add_argument('-i', '--input', required=True, help="Input video file or URL")
    parser.add_argument('-o', '--output', required=True, help="Output video file path")
    parser.add_argument('-t', '--target-lang', choices=LANGUAGE_LIST, default='fr',
                        help="Target language code (choices: %(choices)s, default: %(default)s)")
    parser.add_argument('--model', dest='model', choices=WHISPER_MODELS, default=WHISPER_MODELS[0],
                        help="Whisper model name (choices: %(choices)s, default: %(default)s)")
    parser.add_argument('--tts-model', dest='tts_model', choices=TTS_MODELS, default=TTS_DEFAULT,
                        help="Coqui TTS model name (choices: available models, default: %(default)s)")
    parser.add_argument('--speaker', help="Speaker name to use for multi-speaker TTS models (non-XTTS)")
    parser.add_argument('--speaker-wav', dest='speaker_wav', help="Reference WAV file path for voice cloning models (e.g., XTTS)")
    parser.add_argument('--no-edit', action='store_true', help="Skip interactive text correction")
    parser.add_argument('--edits', help="Path to JSON file with per-segment edits (orig and tgt)")
    parser.add_argument('--preview', action='store_true', help="Play each segment's audio as it's generated and allow early exit")
    parser.add_argument('--segments-file', dest='segments_file',
                        help="Path to JSON file with precomputed segments to skip transcription")
    parser.add_argument('--src-lang', dest='src_lang',
                        help="Source language code (when using --segments-file)")
    parser.add_argument('--accent-language', dest='accent_language',
                        help="Language code to use for TTS accent hack: generate speech in target text with accent of this language (overrides blending and speaker options)")
    parser.add_argument('--lip-sync', action='store_true',
                        help="Apply lip synchronization on the final video using Wav2Lip. Requires --wav2lip-dir and --wav2lip-checkpoint.")
    parser.add_argument('--wav2lip-dir',
                        help="Path to Wav2Lip repository directory containing inference.py")
    parser.add_argument('--wav2lip-checkpoint',
                        help="Path to Wav2Lip checkpoint (.pth) file")
    args = parser.parse_args()
    # Setup and validate lip-sync requirements
    if args.lip_sync:
        # Determine Wav2Lip directory (clone if needed)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_wav2lip_dir = os.path.join(script_dir, 'wav2lip')
        wav2lip_dir = args.wav2lip_dir or default_wav2lip_dir
        if not os.path.isdir(wav2lip_dir):
            print(f"Cloning Wav2Lip into '{wav2lip_dir}'...")
            try:
                subprocess.run(['git', 'clone', 'https://github.com/Rudrabha/Wav2Lip.git', wav2lip_dir], check=True)
            except subprocess.CalledProcessError:
                print("Error: failed to clone Wav2Lip repository", file=sys.stderr)
                sys.exit(1)
        # Check for inference script
        inf = os.path.join(wav2lip_dir, 'inference.py')
        if not os.path.isfile(inf):
            print(f"Error: inference.py not found in Wav2Lip dir '{wav2lip_dir}'", file=sys.stderr)
            sys.exit(1)
        args.wav2lip_dir = wav2lip_dir
        # Determine checkpoint path (download if needed)
        ckpt = args.wav2lip_checkpoint
        if not ckpt:
            ckpt_dir = os.path.join(wav2lip_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = os.path.join(ckpt_dir, 'wav2lip_gan.pth')
        if not os.path.isfile(ckpt):
            # Download checkpoint(s) from Google Drive folder using gdown
            print("Downloading Wav2Lip checkpoint(s) from Google Drive...")
            try:
                import gdown
            except ImportError:
                print("gdown not found, installing via pip...", file=sys.stderr)
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'gdown'], check=True)
                import gdown
            # Google Drive folder ID containing the checkpoint files
            folder_id = "153HLrqlBNxzZcHi17PEvP09kkAfzRshM"
            try:
                gdown.download_folder(id=folder_id, output=ckpt_dir, quiet=False, use_cookies=False)
            except Exception as e:
                print(f"Error: failed to download from Google Drive folder {folder_id}: {e}", file=sys.stderr)
                sys.exit(1)
            # Select a .pth checkpoint if present, else fall back to official GitHub release
            files = os.listdir(ckpt_dir)
            pths = [f for f in files if f.lower().endswith('.pth')]
            if pths:
                selected = pths[0]
                ckpt = os.path.join(ckpt_dir, selected)
                print(f"Using checkpoint: {selected}")
            else:
                # No .pth in Google Drive folder; download official checkpoint
                print("No .pth checkpoint found in Drive; downloading official Wav2Lip checkpoint from GitHub...")
                ckpt = os.path.join(ckpt_dir, 'wav2lip_gan.pth')
                url = 'https://github.com/Rudrabha/Wav2Lip/releases/download/v0.1/wav2lip_gan.pth'
                if shutil.which('curl'):
                    dl_cmd = ['curl', '-L', url, '-o', ckpt]
                elif shutil.which('wget'):
                    dl_cmd = ['wget', url, '-O', ckpt]
                else:
                    print("Error: neither curl nor wget is available; cannot download official Wav2Lip checkpoint.", file=sys.stderr)
                    sys.exit(1)
                try:
                    subprocess.run(dl_cmd, check=True)
                except subprocess.CalledProcessError:
                    print("Error: failed to download official Wav2Lip checkpoint.", file=sys.stderr)
                    sys.exit(1)
                print(f"Downloaded official checkpoint to {ckpt}")
        args.wav2lip_checkpoint = ckpt
        # Ensure minimal Wav2Lip Python dependencies are installed
        deps = []
        try:
            import cv2  # noqa: F401
        except ImportError:
            deps.append('opencv-python')
        try:
            import scipy  # noqa: F401
        except ImportError:
            deps.append('scipy')
        try:
            import tqdm  # noqa: F401
        except ImportError:
            deps.append('tqdm')
        if deps:
            print(f"Installing Wav2Lip dependencies: {deps}")
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + deps, check=True)

    # Load edits JSON if provided (disables interactive editing)
    edits_orig = {}
    edits_tgt = {}
    if args.edits:
        if not os.path.isfile(args.edits):
            print(f"Error: edits file '{args.edits}' not found.", file=sys.stderr)
            sys.exit(1)
        try:
            data = json.load(open(args.edits, encoding='utf-8'))
        except Exception as e:
            print(f"Error: failed to parse edits file: {e}", file=sys.stderr)
            sys.exit(1)
        # Expecting {"orig": {"0": "text", ...}, "tgt": {...}}
        for k, v in data.get('orig', {}).items():
            try:
                edits_orig[int(k)] = v
            except:
                pass
        for k, v in data.get('tgt', {}).items():
            try:
                edits_tgt[int(k)] = v
            except:
                pass
        args.no_edit = True
    tmpdir = tempfile.mkdtemp(prefix="tv_translate_")
    # Step 1: get video
    if args.input.startswith('http://') or args.input.startswith('https://'):
        video_path = download_video(args.input, tmpdir)
    else:
        video_path = args.input
    # Step 2: extract audio
    audio_wav = os.path.join(tmpdir, 'audio.wav')
    extract_audio(video_path, audio_wav)
    # Step 3: load or transcribe segments
    if args.segments_file:
        # Skip transcription; load precomputed segments
        if not os.path.isfile(args.segments_file):
            print(f"Error: segments file '{args.segments_file}' not found.", file=sys.stderr)
            sys.exit(1)
        try:
            seg_data = json.load(open(args.segments_file, encoding='utf-8'))
        except Exception as e:
            print(f"Error: failed to load segments file: {e}", file=sys.stderr)
            sys.exit(1)
        segments = seg_data.get('segments', [])
        # Source language override or fallback to recorded language
        src_lang = seg_data.get('language') or args.src_lang or 'en'
    else:
        # Perform transcription with selected Whisper model
        result = transcribe_audio(audio_wav, args.model)
        segments = result.get('segments', [])
        src_lang = result.get('language', 'en')
    # Step 4: prepare translator (src_lang -> target_lang)
    # We will lazy-load translation per segment
    # Step 5: prepare TTS model selection (must be non-XTTS multilingual or language-specific)
    # Default to user-specified or auto-select
    tts_model = args.tts_model or select_tts_model(args.target_lang)
    # If XTTS model was auto-selected (user did not explicitly set --tts-model), override to a non-XTTS model for language blending
    if args.tts_model is None and 'xtts' in tts_model.lower():
        try:
            from TTS.api import TTS as _TTS
            from TTS.utils.manage import ModelManager
            models_file = _TTS.get_models_file_path()
            mgr = ModelManager(models_file=models_file, progress_bar=False, verbose=False)
            # 1) find language-specific non-XTTS model
            lang_models = [m for m in mgr.list_tts_models() if f"/{args.target_lang}/" in m and 'xtts' not in m]
            if lang_models:
                new_model = lang_models[0]
            else:
                # 2) fallback to any multilingual non-XTTS
                multi_models = [m for m in mgr.list_tts_models() if m.startswith('tts_models/multilingual/') and 'xtts' not in m]
                new_model = multi_models[0] if multi_models else tts_model
            if new_model != tts_model:
                print(f"Warning: XTTS-based model '{tts_model}' overridden to '{new_model}' for language blending", file=sys.stderr)
                tts_model = new_model
        except Exception:
            pass
    # Instantiate TTS API class
    try:
        import torch, importlib, re, builtins
        from TTS.api import TTS
    except Exception as e:
        print(f"Error importing TTS api: {e}", file=sys.stderr)
        sys.exit(1)
    # If accent hack mode enabled but no specific TTS model was set, force multilingual non-XTTS for auto-selection
    if args.accent_language and args.tts_model is None:
        try:
            from TTS.utils.manage import ModelManager
            models_file = TTS.get_models_file_path()
            mgr = ModelManager(models_file=models_file, progress_bar=False, verbose=False)
            multi_models = [m for m in mgr.list_tts_models() if m.startswith('tts_models/multilingual/') and 'xtts' not in m]
            if not multi_models:
                print(f"Error: no multilingual TTS model available for accent hack", file=sys.stderr)
                sys.exit(1)
            new_model = multi_models[0]
            if new_model != tts_model:
                print(f"Info: accent-language hack enabled, switching TTS model from '{tts_model}' to '{new_model}'", file=sys.stderr)
                tts_model = new_model
        except Exception as e2:
            print(f"Warning: accent-language model selection failed: {e2}", file=sys.stderr)
    safe_globals = set()
    while True:
        try:
            tts = TTS(tts_model, progress_bar=False, gpu=False)
            break
        except Exception as e:
            msg = str(e)
            m = re.search(r"Unsupported global: GLOBAL ([^ ]+) was", msg)
            if m:
                name = m.group(1)
                if name in safe_globals:
                    print(f"Error initializing TTS model '{tts_model}': {e}", file=sys.stderr)
                    sys.exit(1)
                try:
                    # Resolve module and attribute; builtins if no module path
                    if '.' in name:
                        mod_name, attr_name = name.rsplit('.', 1)
                        module = importlib.import_module(mod_name)
                    else:
                        module = builtins
                        attr_name = name
                    obj = getattr(module, attr_name)
                    torch.serialization.add_safe_globals([obj])
                    safe_globals.add(name)
                    continue
                except Exception as e2:
                    print(f"Error adding safe global for '{name}': {e2}", file=sys.stderr)
                    sys.exit(1)
            print(f"Error initializing TTS model '{tts_model}': {e}", file=sys.stderr)
            sys.exit(1)
    # Setup voice cloning reference WAV: user-specified or original audio for XTTS
    clone_wav = args.speaker_wav
    if not clone_wav and args.tts_model and 'xtts' in args.tts_model.lower():
        clone_wav = audio_wav
    # Step 6: process segments
    # For per-segment lip-sync, initialize list of output video segments
    if args.lip_sync:
        lip_videos = []
    for i, seg in enumerate(segments):
        # Apply original text edits if provided
        orig_text = edits_orig.get(i, seg.get('text', '').strip())
        # If interactive editing, show segment info and allow edits
        if not args.no_edit:
            # Show segment boundaries and text
            print(f"Segment {i}: [{seg['start']:.2f}s -> {seg['end']:.2f}s]", flush=True)
            print(orig_text, flush=True)
            # 1) Playback prompt: 'p' to play audio, Enter to edit text
            try:
                from pydub import AudioSegment
                from pydub.playback import play
                full_audio = AudioSegment.from_wav(audio_wav)
                start_ms = int(seg['start'] * 1000)
                end_ms = int(seg['end'] * 1000)
            except Exception:
                full_audio = None
            while True:
                choice = input("Press 'p'+Enter to play audio, or Enter to edit text: ").strip().lower()
                if choice == 'p':
                    if full_audio:
                        try:
                            play(full_audio[start_ms:end_ms])
                        except Exception as e:
                            print(f"Error playing audio: {e}", file=sys.stderr)
                    else:
                        print("No audio available to play.", file=sys.stderr)
                    continue
                break
            # 2) Inline edit prompt: prefill with orig_text
            try:
                import readline
                readline.set_startup_hook(lambda: readline.insert_text(orig_text))
            except Exception:
                readline = None
            edited = input('Edit text (or press Enter to keep): ').strip()
            if readline:
                try:
                    readline.set_startup_hook(None)
                except Exception:
                    pass
            if edited:
                orig_text = edited
        # translate (or apply provided translation edits)
        if i in edits_tgt:
            tgt_text = edits_tgt[i]
        else:
            tgt_text = translate_text(orig_text, src_lang, args.target_lang)
        # TTS generation progress with segment text
        total = len(segments)
        print(f"TTS segment {i+1}/{total}: {tgt_text}", flush=True)
        # generate TTS with accent blending via language codes
        seg_wav = os.path.join(tmpdir, f"segment_{i}.wav")
        # Generate TTS audio (supports voice cloning via clone_wav)
        generate_tts_audio(
            tts,
            tgt_text,
            seg_wav,
            speaker=args.speaker,
            speaker_wav=clone_wav,
            language=args.target_lang,
            preview=False,
            accent_language=args.accent_language,
        )
        # per-segment lip synchronization if requested
        if args.lip_sync:
            start = seg['start']
            duration = seg['end'] - start
            seg_video = os.path.join(tmpdir, f"segment_{i}.mp4")
            ffmpeg_cut = [
                'ffmpeg', '-y',
                '-ss', str(start), '-t', str(duration),
                '-i', video_path,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-an', seg_video
            ]
            subprocess.run(ffmpeg_cut, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            seg_lip = os.path.join(tmpdir, f"segment_{i}_lipsync.mp4")
            cmd_lip = [
                sys.executable,
                inf,
                '--checkpoint_path', args.wav2lip_checkpoint,
                '--face', seg_video,
                '--audio', seg_wav,
                '--outfile', seg_lip
            ]
            subprocess.run(cmd_lip, cwd=args.wav2lip_dir, check=True)
            lip_videos.append(seg_lip)
            continue
        # preview translated segment (video + translated audio) if requested
        if args.preview:
            try:
                start = seg['start']
                duration = seg['end'] - start
                preview_file = os.path.join(tmpdir, f"preview_{i}.mp4")
                if not shutil.which('ffmpeg'):
                    print("Warning: ffmpeg not found; cannot generate preview", file=sys.stderr)
                else:
                    # generate a short preview clip (video segment + translated audio)
                    ffmpeg_cmd = [
                        'ffmpeg', '-y',
                        '-ss', str(start), '-t', str(duration),
                        '-i', video_path,
                        '-i', seg_wav,
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                        '-c:a', 'aac',
                        '-map', '0:v:0', '-map', '1:a:0',
                        '-shortest',
                        preview_file
                    ]
                    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"Preview segment {i}: {preview_file}")
                    # play preview using ffplay or system default opener
                    # play preview using available video player
                    played = False
                    # try ffplay
                    if shutil.which('ffplay'):
                        subprocess.run([
                            'ffplay', '-hide_banner', '-loglevel', 'error',
                            '-autoexit', preview_file
                        ])
                        played = True
                    # try mpv
                    elif shutil.which('mpv'):
                        subprocess.run([
                            'mpv', '--really-quiet', '--no-video-title-show', preview_file
                        ])
                        played = True
                    # try mplayer
                    elif shutil.which('mplayer'):
                        subprocess.run([
                            'mplayer', '-quiet', preview_file
                        ])
                        played = True
                    # fallback to system opener
                    if not played:
                        opened = False
                        for opener in ('xdg-open', 'open', 'start'):
                            if shutil.which(opener):
                                subprocess.run([opener, preview_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                opened = True
                                break
                        if not opened:
                            print(f"Preview saved to {preview_file}")
            except subprocess.CalledProcessError:
                print("Error generating or playing preview segment", file=sys.stderr)
            except Exception as e:
                print(f"Error in preview mode: {e}", file=sys.stderr)
    # Step 7: output final video
    if args.lip_sync:
        # concatenate all lip-synced segment videos
        concat_list = os.path.join(tmpdir, 'lip_segments.txt')
        with open(concat_list, 'w') as f:
            for vp in lip_videos:
                f.write(f"file '{vp}'\n")
        cmd_concat = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_list,
            '-c', 'copy', args.output
        ]
        try:
            subprocess.run(cmd_concat, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: failed to concatenate lip-synced segments: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        video_duration = segments[-1]['end'] if segments else 0
        combined_audio = os.path.join(tmpdir, 'combined.wav')
        combine_audio_segments(segments, tmpdir, video_duration, combined_audio)
        merge_video_audio(video_path, combined_audio, args.output)
    print(f"Output video saved to {args.output}", flush=True)

if __name__ == '__main__':
    main()