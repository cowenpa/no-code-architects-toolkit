# services/v1/media/media_transcribe.py
# SPDX-License-Identifier: GPL-2.0+
#
# Updated 2025-05-17 by Paul & ChatGPT
# • Skip HTTP download for local files under /media/upload
# • Resolve internal HTTP URLs to local paths
# • Rename source video to *_transcribed.mp4 when done
# ---------------------------------------------------------------------

import os
import re
import logging
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse, unquote

import whisper
import srt
from datetime import timedelta
from whisper.utils import WriteSRT, WriteVTT            # still used downstream
from services.file_management import download_file      # unchanged
from config import LOCAL_STORAGE_PATH

# ------------------------------------------------------------------ #
#  CONFIGURATION CONSTANTS                                           #
# ------------------------------------------------------------------ #

HOSTNAME       = "ncatoolkit.kingdomautomations.com"    # your domain (edit if different)
UPLOAD_HOST    = Path("/srv/media/upload")              # host path (bind-mounted RO)
UPLOAD_CONT    = Path("/app/media")                     # same folder INSIDE container
logger         = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------ #
#  HELPER:  resolve media_url → local Path (download only if needed) #
# ------------------------------------------------------------------ #
def resolve_media_path(media_url: str) -> tuple[Path, bool]:
    """
    Return (local_path, downloaded_flag).

    • Accepts absolute file paths ( /media/upload/… or /srv/media/upload/… ).
    • Converts internal HTTP URLs to local paths.
    • Falls back to download_file() for true remote URLs.
    """
    # 1) Caller passed a direct, existing path
    p = Path(media_url)
    if p.is_absolute() and p.exists():
        logger.info(f"Using local media file: {p}")
        return p, False

    # 2) Internal HTTP URL → map /upload/<file> → UPLOAD_HOST / <file>
    parts = urlparse(media_url)
    if parts.scheme in {"http", "https"} and parts.hostname == HOSTNAME:
        try:
            rel = PurePosixPath(unquote(parts.path)).relative_to("/upload")
            # Try all possible mount locations
            for prefix in [UPLOAD_HOST, UPLOAD_CONT]:
                local_file = prefix / rel
                if local_file.exists():
                    logger.info(f"Resolved internal URL to local file: {local_file}")
                    return local_file, False
        except ValueError:
            pass

    # 3) Remote URL – download
    dl_path = download_file(media_url, os.path.join(LOCAL_STORAGE_PATH, "downloads"))
    logger.info(f"Downloaded remote media to: {dl_path}")
    return Path(dl_path), True

# ------------------------------------------------------------------ #
#  MAIN ENTRY                                                        #
# ------------------------------------------------------------------ #
def process_transcribe_media(
        media_url: str,
        task: str,
        include_text: bool,
        include_srt: bool,
        include_segments: bool,
        word_timestamps: bool,
        response_type: str,
        language: str,
        job_id: str,
words_per_line: int | None = None
):
    """
    Transcribe (or translate) media_url with Whisper.
    Supports local files in /media/upload and remote URLs.
    Renames the source video to *_transcribed.mp4 after success.
    """
    logger.info(f"Starting {task} for media URL: {media_url}")

    # ------------------------------------------------------------------ #
    #  1.  Locate or download the media                                  #
    # ------------------------------------------------------------------ #
    input_path, downloaded = resolve_media_path(media_url)

    # ------------------------------------------------------------------ #
    #  2.  Run Whisper                                                   #
    # ------------------------------------------------------------------ #
    model_size = "base"
    model = whisper.load_model(model_size, device="cpu")
    logger.info(f"Loaded Whisper {model_size} model (CPU/FP32)")

    options = {
        "task": task,
        "word_timestamps": word_timestamps,
        "verbose": False
    }
    if language:
        options["language"] = language

    result = model.transcribe(str(input_path), **options)
    logger.info(f"Whisper {task} completed")

    # ------------------------------------------------------------------ #
    #  3.  Prepare outputs (unchanged code, trimmed for brevity)         #
    # ------------------------------------------------------------------ #
    text = srt_text = segments_json = None
    # ... (keep your original subtitle-building logic here) ...

    # ------------------------------------------------------------------ #
    #  4.  Clean-up downloaded file                                      #
    # ------------------------------------------------------------------ #
    if downloaded and input_path.exists():
        input_path.unlink(missing_ok=True)
        logger.info(f"Removed temp download: {input_path}")

    # ------------------------------------------------------------------ #
    #  5.  Rename original media to *_transcribed.mp4                    #
    # ------------------------------------------------------------------ #
    try:
        # Only rename if the file still exists (i.e., wasn’t deleted)
        if input_path.exists():
            new_name = input_path.with_name(
                input_path.stem + "_transcribed" + input_path.suffix
            )
            input_path.rename(new_name)
            logger.info(f"Renamed source video → {new_name.name}")
    except Exception as e:
        logger.warning(f"Could not rename source video: {e}")

    # ------------------------------------------------------------------ #
    #  6.  Return or save artifacts (same as before)                     #
    # ------------------------------------------------------------------ #
    if response_type == "direct":
        return text, srt_text, segments_json

    # save to disk (unchanged from your original implementation) ...
    # text_filename, srt_filename, segments_filename = ...

    return text_filename, srt_filename, segments_filename
