#!/usr/bin/env python3
"""
translate_video_worker.py: Worker script to download, extract audio, and transcribe video segments.
"""
import sys
import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Worker: download and transcribe video segments")
    parser.add_argument('-i', '--input', required=True, help='Video URL or file path')
    parser.add_argument('--model', required=True, help='Whisper model name')
    parser.add_argument('--tmpdir', required=True, help='Temporary directory to use')
    args = parser.parse_args()

    # import functions from translate_video CLI
    try:
        from translate_video import download_video, extract_audio, transcribe_audio
    except ImportError:
        # adjust path if needed
        sys.path.append(os.path.dirname(__file__))
        from translate_video import download_video, extract_audio, transcribe_audio

    # Step 1: download video
    video_path = download_video(args.input, args.tmpdir)

    # Step 2: extract audio
    audio_path = os.path.join(args.tmpdir, 'audio.wav')
    extract_audio(video_path, audio_path)

    # Step 3: transcribe audio
    result = transcribe_audio(audio_path, args.model)
    # Capture detected source language and segments list
    src_lang = result.get('language', 'en')
    segments = result.get('segments', [])

    # Write segments and language to JSON file
    out_file = os.path.join(args.tmpdir, 'segments.json')
    data = {'language': src_lang, 'segments': segments}
    try:
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # Notify GUI of JSON path
        sys.stdout.write(f'JSONWRITTEN:{out_file}\n')
        sys.stdout.flush()
    except Exception as e:
        sys.stderr.write(f'Error writing segments.json: {e}\n')
        sys.exit(1)

    # Stream each segment as it's available
    for i, seg in enumerate(segments):
        text = seg.get('text', '').strip()
        sys.stdout.write(f'SEGMENT:{i}:{text}\n')
        sys.stdout.flush()
    # Indicate completion
    sys.stdout.write('Done\n')
    sys.stdout.flush()

if __name__ == '__main__':
    main()