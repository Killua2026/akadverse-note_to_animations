"""AkadVerse Note-to-Animations AI v3.0.

NotebookLM-style explainer pipeline:
- Multipart input: typed text + optional PDF
- Format control: explainer or brief
- Theme control: fixed palette + custom override
- Contextual illustrations via pluggable visual provider
- Single voice narration synced to visuals
"""

import asyncio
import base64
import binascii
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import edge_tts
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from langchain_community.document_loaders import PyPDFLoader

    PDF_LOADER_AVAILABLE = True
except ImportError:
    PyPDFLoader = None  # type: ignore[assignment]
    PDF_LOADER_AVAILABLE = False

try:
    from google import genai
    from google.genai import types

    GOOGLE_SDK_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    GOOGLE_SDK_AVAILABLE = False

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NoteToAnimationsAI")

OUTPUT_DIR = "generated_animations"
TEMP_DIR = "animation_temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

JOBS: Dict[str, Dict[str, Any]] = {}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_SOURCE_CHARS = 20000


FORMAT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "explainer": {
        "segment_count": 8,
        "segment_duration": 22,
        "visual_density": "high",
        "tone": "deep_dive",
    },
    "brief": {
        "segment_count": 4,
        "segment_duration": 14,
        "visual_density": "medium",
        "tone": "highlight_summary",
    },
}


THEME_PRESETS: Dict[str, Dict[str, str]] = {
    "classic_whiteboard": {
        "style_prompt": "clean whiteboard sketch, educational markers, simple shapes",
        "fallback_color": "white",
    },
    "watercolor": {
        "style_prompt": "soft watercolor educational painting, airy palette",
        "fallback_color": "lightyellow",
    },
    "anime": {
        "style_prompt": "polished anime educational scene, clear cinematic composition",
        "fallback_color": "lightskyblue",
    },
    "papercraft": {
        "style_prompt": "layered paper cutout illustration, tactile classroom style",
        "fallback_color": "wheat",
    },
    "retro_print": {
        "style_prompt": "retro print poster style, halftone educational textures",
        "fallback_color": "moccasin",
    },
}


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    cleaned = re.sub(r"```json\s*|\s*```", "", text or "").strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError as error:
        logger.warning("JSON parse warning: %s", error)
        return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text or "").strip()


def _clip_text(text: str, max_chars: int = MAX_SOURCE_CHARS) -> str:
    cleaned = _normalize_text(text)
    return cleaned[:max_chars]


def _first_sentences(text: str, count: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", _normalize_text(text))
    picked = [segment.strip() for segment in sentences if segment.strip()][:count]
    if picked:
        return " ".join(picked)
    return "Key concept summary unavailable from source."


def _extract_keyword_phrases(text: str, max_items: int = 6) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", text.lower())
    ignore = {
        "that",
        "this",
        "with",
        "from",
        "were",
        "have",
        "about",
        "their",
        "they",
        "into",
        "which",
        "will",
        "your",
        "there",
        "where",
        "what",
        "when",
        "them",
        "then",
        "than",
        "because",
        "while",
        "students",
        "lecture",
        "course",
    }
    freq: Dict[str, int] = {}
    for word in words:
        if word in ignore:
            continue
        freq[word] = freq.get(word, 0) + 1

    ranked = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    return [word for word, _ in ranked[:max_items]]


def _default_storyboard(
    source_text: str,
    subject: str,
    format_type: str,
    visual_theme: str,
    segment_count: int,
    segment_duration: int,
) -> Dict[str, Any]:
    title = subject.strip() or "Explainer Overview"
    keywords = _extract_keyword_phrases(source_text)
    seed_summary = _first_sentences(source_text, count=3)

    segments: List[Dict[str, Any]] = []
    for index in range(segment_count):
        focus = keywords[index % max(len(keywords), 1)] if keywords else f"concept_{index+1}"
        narration = (
            f"Section {index + 1}. We focus on {focus}. "
            f"{seed_summary}"
        )
        segments.append(
            {
                "segment_index": index + 1,
                "slide_title": f"{title}: Part {index + 1}",
                "narration": narration,
                "visual_prompt": (
                    f"Educational illustration for {focus} in {visual_theme} style. "
                    "No text, no numbers, no labels, no watermarks."
                ),
                "duration_seconds": segment_duration,
            }
        )

    return {
        "title": title,
        "format": format_type,
        "theme": visual_theme,
        "overview": seed_summary,
        "segments": segments,
    }


def get_valid_model_name(api_key_str: Optional[str], purpose: str = "text") -> str:
    """Discover a suitable Gemini model instead of hardcoding a single value."""
    if not GOOGLE_SDK_AVAILABLE or not api_key_str or genai is None:
        return "unavailable"
    try:
        client = genai.Client(api_key=api_key_str)
        available = [m.name.replace("models/", "") for m in client.models.list() if m.name]
        if not available:
            return "unavailable"

        if purpose == "image":
            image_priority = [
                "gemini-3-flash-image",
                "gemini-2.0-flash-preview-image-generation",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
            ]
            for preferred in image_priority:
                if preferred in available:
                    return preferred
            for discovered in available:
                if "image" in discovered:
                    return discovered
            return available[0]

        text_priority = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        for preferred in text_priority:
            if preferred in available:
                return preferred
        return available[0]
    except Exception as error:
        logger.warning("Model discovery failed: %s", error)
        return "unavailable"


ACTIVE_TEXT_MODEL = get_valid_model_name(GOOGLE_API_KEY, purpose="text")
ACTIVE_IMAGE_MODEL = get_valid_model_name(GOOGLE_API_KEY, purpose="image")


async def _run_subprocess(cmd: List[str], check: bool = True) -> Tuple[bool, str]:
    def _runner() -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(cmd, capture_output=True, check=check)

    try:
        result = await asyncio.to_thread(_runner)
        return True, (result.stderr or b"").decode(errors="ignore")
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or b"").decode(errors="ignore")
        return False, stderr
    except Exception as error:
        return False, str(error)


async def _extract_pdf_text(file_path: str) -> str:
    loader_cls = PyPDFLoader
    if not PDF_LOADER_AVAILABLE or loader_cls is None:
        logger.warning("PDF loader unavailable. Install langchain-community and pypdf.")
        return ""

    def _loader() -> str:
        loader = loader_cls(file_path)
        pages = loader.load()
        return "\n\n".join(page.page_content for page in pages)

    try:
        extracted = await asyncio.to_thread(_loader)
        return _normalize_text(extracted)
    except Exception as error:
        logger.error("PDF extraction failed: %s", error)
        return ""


def _build_storyboard_prompt(
    content: str,
    subject: str,
    format_type: str,
    visual_theme: str,
    style_prompt: str,
    segment_count: int,
    segment_duration: int,
    visual_density: str,
    tone: str,
) -> str:
    return f"""
You are AkadVerse Creative Director, with three active perspectives:
- Leo: systems architect
- Sarah: ML engineer
- Dr. Adeyemi: pedagogy expert

Convert source material into a NotebookLM-style explainer video storyboard.

Constraints:
- Format type: {format_type}
- Segment count target: {segment_count}
- Segment duration target: {segment_duration} seconds
- Visual density: {visual_density}
- Teaching tone: {tone}
- Theme: {visual_theme}
- Theme style direction: {style_prompt}
- A single narrator voice should speak all segments.
- Visual prompts must never ask for words, letters, digits, UI text, or watermark.

Return ONLY valid JSON with this schema:
{{
  "title": "string",
  "overview": "string",
  "segments": [
    {{
      "segment_index": 1,
      "slide_title": "string",
      "narration": "string",
      "visual_prompt": "string",
      "duration_seconds": 20
    }}
  ]
}}

Subject:
{subject}

Source notes:
{content}
""".strip()


def _validate_storyboard(storyboard: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not storyboard:
        return fallback

    segments = storyboard.get("segments")
    if not isinstance(segments, list) or not segments:
        return fallback

    normalized_segments: List[Dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            continue
        narration = str(segment.get("narration", "")).strip()
        visual_prompt = str(segment.get("visual_prompt", "")).strip()
        slide_title = str(segment.get("slide_title", "")).strip() or f"Segment {index}"
        try:
            duration_seconds = int(segment.get("duration_seconds", 20))
        except (TypeError, ValueError):
            duration_seconds = 20
        duration_seconds = max(8, min(duration_seconds, 80))

        if not narration:
            continue
        if not visual_prompt:
            visual_prompt = "Educational symbolic illustration with no text"

        normalized_segments.append(
            {
                "segment_index": index,
                "slide_title": slide_title,
                "narration": narration,
                "visual_prompt": visual_prompt,
                "duration_seconds": duration_seconds,
            }
        )

    if not normalized_segments:
        return fallback

    return {
        "title": str(storyboard.get("title", fallback.get("title", "Explainer"))).strip() or "Explainer",
        "overview": str(storyboard.get("overview", fallback.get("overview", ""))).strip(),
        "segments": normalized_segments,
    }


async def generate_storyboard(
    source_text: str,
    subject: str,
    format_type: str,
    visual_theme: str,
    style_prompt: str,
) -> Dict[str, Any]:
    config = FORMAT_CONFIGS.get(format_type, FORMAT_CONFIGS["explainer"])
    fallback = _default_storyboard(
        source_text=source_text,
        subject=subject,
        format_type=format_type,
        visual_theme=visual_theme,
        segment_count=config["segment_count"],
        segment_duration=config["segment_duration"],
    )

    if not GOOGLE_SDK_AVAILABLE or not GOOGLE_API_KEY or genai is None:
        return fallback

    prompt = _build_storyboard_prompt(
        content=_clip_text(source_text, max_chars=12000),
        subject=subject,
        format_type=format_type,
        visual_theme=visual_theme,
        style_prompt=style_prompt,
        segment_count=config["segment_count"],
        segment_duration=config["segment_duration"],
        visual_density=config["visual_density"],
        tone=config["tone"],
    )

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=ACTIVE_TEXT_MODEL,
            contents=prompt,
        )
        parsed = _safe_json_loads(getattr(response, "text", "") or "")
        if not parsed:
            return fallback
        return _validate_storyboard(parsed, fallback)
    except Exception as error:
        logger.error("Storyboard generation failed: %s", error)
        return fallback


class VisualProvider:
    async def generate_image(self, prompt: str, output_path: str) -> bool:
        raise NotImplementedError


def _extract_image_bytes_from_response(response: Any) -> Optional[bytes]:
    def _coerce_bytes(payload: Any) -> Optional[bytes]:
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        if isinstance(payload, str):
            try:
                return base64.b64decode(payload, validate=True)
            except (binascii.Error, ValueError):
                return payload.encode("utf-8")
        return None

    def _is_image_mime(mime_type: Optional[str]) -> bool:
        return bool(mime_type and mime_type.lower().startswith("image/"))

    # Primary path for google-genai==1.2.0:
    # response.candidates[].content.parts[].inline_data.{mime_type,data}
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list):
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None)
            if not isinstance(parts, list):
                continue
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data is None:
                    continue
                mime_type = getattr(inline_data, "mime_type", None)
                if not _is_image_mime(mime_type):
                    continue
                payload = getattr(inline_data, "data", None)
                coerced = _coerce_bytes(payload)
                if coerced:
                    return coerced

    # Compatibility path for alternate wrappers.
    direct_payload = getattr(response, "image_bytes", None)
    coerced_direct = _coerce_bytes(direct_payload)
    if coerced_direct:
        return coerced_direct

    return None


class GeminiImageProvider(VisualProvider):
    def __init__(self, api_key: Optional[str], model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    async def generate_image(self, prompt: str, output_path: str) -> bool:
        if not self.api_key or not GOOGLE_SDK_AVAILABLE or genai is None or types is None:
            return False

        try:
            client = genai.Client(api_key=self.api_key)
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_name,
                contents=(
                    "Create a single educational illustration. "
                    "No text, no letters, no numbers, no watermark. "
                    + prompt
                ),
                config=types.GenerateContentConfig(response_modalities=[types.Modality.IMAGE]),
            )

            image_bytes = _extract_image_bytes_from_response(response)
            if not image_bytes:
                logger.warning("Gemini image response did not contain image bytes")
                return False

            def _write() -> None:
                with open(output_path, "wb") as file_obj:
                    file_obj.write(image_bytes)

            await asyncio.to_thread(_write)
            return True
        except Exception as error:
            logger.error("Gemini image generation failed: %s", error)
            return False


async def generate_fallback_image(output_path: str, color: str = "white") -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=1280x720",
        "-frames:v",
        "1",
        output_path,
    ]
    success, stderr = await _run_subprocess(cmd, check=False)
    if not success:
        logger.error("Fallback image generation failed: %s", stderr)
    return success


def pick_voice_for_theme(theme_name: str) -> str:
    theme_voice_map = {
        "classic_whiteboard": "en-US-AvaNeural",
        "watercolor": "en-US-JennyNeural",
        "anime": "en-US-GuyNeural",
        "papercraft": "en-US-EmmaNeural",
        "retro_print": "en-GB-SoniaNeural",
    }
    return theme_voice_map.get(theme_name, "en-US-AvaNeural")


async def synthesize_audio(text: str, output_path: str, voice: str) -> bool:
    try:
        communicator = edge_tts.Communicate(text, voice)
        await communicator.save(output_path)
        return True
    except Exception as error:
        logger.error("TTS synthesis failed: %s", error)
        return False


async def generate_silent_audio(output_path: str, duration_seconds: int) -> bool:
    safe_duration = max(4, min(duration_seconds, 120))
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t",
        str(safe_duration),
        "-q:a",
        "9",
        "-acodec",
        "libmp3lame",
        output_path,
    ]
    success, stderr = await _run_subprocess(cmd, check=False)
    if not success:
        logger.error("Silent audio fallback failed: %s", stderr)
    return success


async def create_segment_video(image_path: str, audio_path: str, output_path: str) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        image_path,
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=1280:720,format=yuv420p",
        "-shortest",
        output_path,
    ]
    success, stderr = await _run_subprocess(cmd, check=False)
    if not success:
        logger.error("Segment render failed: %s", stderr)
    return success


async def assemble_video(segments: List[str], output_path: str) -> bool:
    if not segments:
        return False
    if len(segments) == 1:
        try:
            await asyncio.to_thread(shutil.copy, segments[0], output_path)
            return True
        except Exception as error:
            logger.error("Single segment copy failed: %s", error)
            return False

    concat_manifest = output_path + ".txt"

    def _write_manifest() -> None:
        with open(concat_manifest, "w", encoding="utf-8") as manifest:
            for seg_path in segments:
                normalized = seg_path.replace("'", "'\\''")
                manifest.write(f"file '{normalized}'\n")

    await asyncio.to_thread(_write_manifest)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_manifest,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        output_path,
    ]
    success, stderr = await _run_subprocess(cmd, check=False)
    if not success:
        logger.error("Final assembly failed: %s", stderr)

    try:
        await asyncio.to_thread(os.remove, concat_manifest)
    except OSError:
        pass

    return success


def _resolve_theme(visual_theme: str, custom_theme_style: Optional[str]) -> Tuple[str, Dict[str, str]]:
    theme_key = visual_theme.strip().lower()
    if theme_key in THEME_PRESETS:
        theme_data = dict(THEME_PRESETS[theme_key])
    else:
        theme_key = "classic_whiteboard"
        theme_data = dict(THEME_PRESETS[theme_key])

    if custom_theme_style:
        theme_data["style_prompt"] = custom_theme_style.strip()

    return theme_key, theme_data


async def process_animation_job(
    job_id: str,
    source_text: str,
    subject: str,
    format_type: str,
    visual_theme: str,
    custom_theme_style: Optional[str],
) -> None:
    job = JOBS[job_id]
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    theme_key, theme_data = _resolve_theme(visual_theme, custom_theme_style)
    visual_provider: VisualProvider = GeminiImageProvider(GOOGLE_API_KEY, ACTIVE_IMAGE_MODEL)

    try:
        job["status"] = "planning"
        job["progress"] = 10
        job["message"] = "Building explainer storyboard"

        storyboard = await generate_storyboard(
            source_text=source_text,
            subject=subject,
            format_type=format_type,
            visual_theme=theme_key,
            style_prompt=theme_data["style_prompt"],
        )
        segments = storyboard.get("segments", [])
        if not segments:
            raise RuntimeError("Storyboard generation returned no segments")

        job["storyboard_title"] = storyboard.get("title", "Explainer")
        job["segment_count"] = len(segments)

        voice = pick_voice_for_theme(theme_key)
        rendered_segments: List[str] = []

        for index, segment in enumerate(segments, start=1):
            job["status"] = "rendering_segments"
            phase_progress = int((index / len(segments)) * 75)
            job["progress"] = 15 + phase_progress
            job["message"] = f"Rendering segment {index}/{len(segments)}"

            image_path = os.path.join(job_dir, f"image_{index}.png")
            audio_path = os.path.join(job_dir, f"audio_{index}.mp3")
            segment_path = os.path.join(job_dir, f"segment_{index}.mp4")

            prompt = f"{segment.get('visual_prompt', '')}. Theme direction: {theme_data['style_prompt']}"
            image_ok = await visual_provider.generate_image(prompt, image_path)
            if not image_ok:
                await generate_fallback_image(image_path, theme_data["fallback_color"])

            narration_text = str(segment.get("narration", "")).strip()
            duration_seconds = int(segment.get("duration_seconds", 20))
            audio_ok = await synthesize_audio(narration_text, audio_path, voice)
            if not audio_ok:
                await generate_silent_audio(audio_path, duration_seconds)

            segment_ok = await create_segment_video(image_path, audio_path, segment_path)
            if not segment_ok:
                raise RuntimeError(f"Segment {index} render failed")
            rendered_segments.append(segment_path)

        job["status"] = "finalizing"
        job["progress"] = 95
        job["message"] = "Assembling final video"

        final_filename = f"{job_id}_animation.mp4"
        final_path = os.path.join(OUTPUT_DIR, final_filename)
        assembled = await assemble_video(rendered_segments, final_path)
        if not assembled:
            raise RuntimeError("Final assembly failed")

        job["status"] = "completed"
        job["progress"] = 100
        job["message"] = "Explainer ready"
        job["download_url"] = f"/animations/download/{final_filename}"
    except Exception as error:
        logger.error("Animation job failed [%s]: %s", job_id, error)
        job["status"] = "failed"
        job["error"] = "Service degraded gracefully. Unable to complete rendering for this request."
        job["internal_error"] = str(error)
    finally:
        try:
            await asyncio.to_thread(shutil.rmtree, job_dir, True)
        except Exception as cleanup_error:
            logger.warning("Cleanup warning for job %s: %s", job_id, cleanup_error)


class ThemeInfo(BaseModel):
    name: str
    style_prompt: str


app = FastAPI(title="AkadVerse Note-to-Animations AI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/animations/generate")
async def generate_animation(
    tasks: BackgroundTasks,
    content: str = Form(default=""),
    subject: str = Form(default=""),
    format_type: str = Form(default="explainer", alias="format"),
    visual_theme: str = Form(default="classic_whiteboard"),
    custom_theme_style: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
) -> Dict[str, Any]:
    normalized_format = format_type.strip().lower()
    if normalized_format not in FORMAT_CONFIGS:
        raise HTTPException(status_code=400, detail="Invalid format. Use explainer or brief.")

    typed_text = _normalize_text(content)
    source_parts: List[str] = []
    source_type = "text"

    if typed_text:
        source_parts.append(typed_text)

    if file is not None:
        filename = (file.filename or "").lower()
        if not filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported on this endpoint.")

        source_type = "text_plus_pdf" if typed_text else "pdf"
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                tmp_path = temp_file.name
                data = await file.read()
                temp_file.write(data)

            pdf_text = await _extract_pdf_text(tmp_path)
            if pdf_text:
                source_parts.append(pdf_text)
        except Exception as error:
            logger.error("PDF intake error: %s", error)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as error:
                    logger.warning("Temporary file cleanup warning: %s", error)

    merged_source = _clip_text("\n\n".join(part for part in source_parts if part.strip()))
    if not merged_source:
        raise HTTPException(status_code=400, detail="Provide typed content or upload a PDF source.")

    job_id = str(uuid.uuid4())[:8].upper()
    theme_key, _ = _resolve_theme(visual_theme, custom_theme_style)
    JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "created_at": datetime.now().isoformat(),
        "source_type": source_type,
        "format": normalized_format,
        "visual_theme": theme_key,
        "subject": subject.strip() or "",
    }

    tasks.add_task(
        process_animation_job,
        job_id,
        merged_source,
        subject.strip(),
        normalized_format,
        visual_theme,
        custom_theme_style,
    )
    return {
        "job_id": job_id,
        "poll_url": f"/animations/status/{job_id}",
        "source_type": source_type,
        "format": normalized_format,
        "visual_theme": theme_key,
    }


@app.get("/animations/themes", response_model=List[ThemeInfo])
async def list_themes() -> List[ThemeInfo]:
    return [
        ThemeInfo(name=name, style_prompt=data["style_prompt"])
        for name, data in THEME_PRESETS.items()
    ]


@app.get("/animations/status/{job_id}")
async def get_status(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/animations/download/{filename}")
async def download(filename: str) -> FileResponse:
    output_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Rendered file not found")
    return FileResponse(output_path, media_type="video/mp4", filename=filename)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "script_model": ACTIVE_TEXT_MODEL,
        "image_model": ACTIVE_IMAGE_MODEL,
        "audio_engine": "edge-tts",
        "visual_provider": "gemini_image_provider",
        "google_sdk_available": GOOGLE_SDK_AVAILABLE,
    }


if __name__ == "__main__":
    uvicorn.run("note_to_animations:app", host="127.0.0.1", port=8015, reload=True)