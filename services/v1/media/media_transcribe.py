import os
import whisper
import srt
from datetime import timedelta
from whisper.utils import WriteSRT, WriteVTT
from services.file_management import download_file
import logging
from config import LOCAL_STORAGE_PATH
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse, unquote

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define your domain/hostnames as needed for local file mapping
HOSTNAME = "ncatoolkit.kingdomautomations.com"
UPLOAD_HOST = Path("/srv/media/upload")
UPLOAD_CONT = Path("/app/media")

def resolve_media_path(media_url):
    """
    Accepts:
      - Absolute file paths that exist locally
      - Internal URLs like http(s)://ncatoolkit.kingdomautomations.com/upload/file.mp4 (mapped to local)
      - Remote URLs (downloads them to temp file as before)
    Returns:
      - path (str) to use as input to Whisper
      - flag: True if file was downloaded and should be deleted after use
    """
    # 1. Absolute local file?
    p = Path(media_url)
    if p.is_absolute() and p.exists():
        logger.info(f"Using local media file: {p}")
        return str(p), False

    # 2. Internal HTTP(S) URL for our known hostname?
    parts = urlparse(media_url)
    if parts.scheme in {"http", "https"} and parts.hostname == HOSTNAME:
        try:
            rel = PurePosixPath(unquote(parts.path)).relative_to("/upload")
            # Try both bind mount (host) and container mount
            for prefix in [UPLOAD_HOST, UPLOAD_CONT]:
                local_file = prefix / rel
                if local_file.exists():
                    logger.info(f"Resolved internal URL to local file: {local_file}")
                    return str(local_file), False
        except Exception:
            pass

    # 3. Remote URL (fallback: download as before)
    dl_path = download_file(media_url, os.path.join(LOCAL_STORAGE_PATH, f"{os.urandom(8).hex()}_input"))
    logger.info(f"Downloaded remote media to: {dl_path}")
    return dl_path, True

def process_transcribe_media(media_url, task, include_text, include_srt, include_segments, word_timestamps, response_type, language, job_id, words_per_line=None):
    """Transcribe or translate media and return the transcript/translation, SRT or VTT file path."""
    logger.info(f"Starting {task} for media URL: {media_url}")
    input_filename, downloaded = resolve_media_path(media_url)
    logger.info(f"Resolved media file to: {input_filename}")

    try:
        model_size = "base"
        model = whisper.load_model(model_size)
        logger.info(f"Loaded Whisper {model_size} model")

        options = {
            "task": task,
            "word_timestamps": word_timestamps,
            "verbose": False
        }
        if language:
            options["language"] = language

        result = model.transcribe(input_filename, **options)
        text = None
        srt_text = None
        segments_json = None

        logger.info(f"Generated {task} output")

        if include_text is True:
            text = result['text']

        if include_srt is True:
            srt_subtitles = []
            subtitle_index = 1

            if words_per_line and words_per_line > 0:
                all_words = []
                word_timings = []
                for segment in result['segments']:
                    words = segment['text'].strip().split()
                    segment_start = segment['start']
                    segment_end = segment['end']
                    if words:
                        duration_per_word = (segment_end - segment_start) / len(words)
                        for i, word in enumerate(words):
                            word_start = segment_start + (i * duration_per_word)
                            word_end = word_start + duration_per_word
                            all_words.append(word)
                            word_timings.append((word_start, word_end))
                current_word = 0
                while current_word < len(all_words):
                    chunk = all_words[current_word:current_word + words_per_line]
                    chunk_start = word_timings[current_word][0]
                    chunk_end = word_timings[min(current_word + len(chunk) - 1, len(word_timings) - 1)][1]
                    srt_subtitles.append(srt.Subtitle(
                        subtitle_index,
                        timedelta(seconds=chunk_start),
                        timedelta(seconds=chunk_end),
                        ' '.join(chunk)
                    ))
                    subtitle_index += 1
                    current_word += words_per_line
            else:
                for segment in result['segments']:
                    start = timedelta(seconds=segment['start'])
                    end = timedelta(seconds=segment['end'])
                    segment_text = segment['text'].strip()
                    srt_subtitles.append(srt.Subtitle(subtitle_index, start, end, segment_text))
                    subtitle_index += 1

            srt_text = srt.compose(srt_subtitles)

        if include_segments is True:
            segments_json = result['segments']

        # Clean up downloaded temp file (if any)
        if downloaded and os.path.exists(input_filename):
            os.remove(input_filename)
            logger.info(f"Removed temp file: {input_filename}")

        logger.info(f"{task.capitalize()} successful, output type: {response_type}")

        if response_type == "direct":
            return text, srt_text, segments_json
        else:
            if include_text is True:
                text_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.txt")
                with open(text_filename, 'w') as f:
                    f.write(text)
            else:
                text_file = None

            if include_srt is True:
                srt_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.srt")
                with open(srt_filename, 'w') as f:
                    f.write(srt_text)
            else:
                srt_filename = None

            if include_segments is True:
                segments_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.json")
                with open(segments_filename, 'w') as f:
                    f.write(str(segments_json))
            else:
                segments_filename = None

            return text_filename, srt_filename, segments_filename 

    except Exception as e:
        logger.error(f"{task.capitalize()} failed: {str(e)}")
        raise