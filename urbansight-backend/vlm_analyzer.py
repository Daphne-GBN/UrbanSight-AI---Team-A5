
"""
UrbanSight AI - Vision Language Model Analyzer
Uses Qwen2.5-VL 3B through the local Ollama API.
Only anonymized images should be passed to this module.
"""

import base64
import json
import urllib.request
import urllib.error


MODEL_NAME = "qwen2.5vl:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"


SYSTEM_PROMPT = """
You are UrbanSight AI, a street-scene observation assistant.

Analyze the provided anonymized street photograph.

RULES:
1. Describe only objects and features visibly supported by the image.
2. Never identify people or infer their identity, age, or occupation.
3. Never read, reproduce, or infer vehicle license plate numbers.
4. Do not estimate vehicle speed or movement from a still image.
5. Do not invent road markings, signs, infrastructure, or road conditions.
6. If something cannot be determined, explicitly state that.
7. Distinguish observations from potential safety concerns.
8. Use the supplied vehicle detection counts as additional context,
   not as proof of other scene features.
9. Do not claim privacy blurring is complete or guaranteed.

Return valid JSON only, using this structure:

{
    "scene_description": "",
    "road_features": {
        "road_type": "",
        "markings": [],
        "road_condition": ""
    },
    "traffic_observations": {
        "vehicles": [],
        "traffic_density": "Low/Moderate/Heavy/Unknown",
        "pedestrian_presence": ""
    },
    "infrastructure": [],
    "safety_observations": [],
    "limitations": []
}
"""


def generate_scene_summary(
    user_prompt: str,
    detection_summary: dict,
    faces_blurred: int,
    plates_blurred: int,
    image_bytes: bytes
) -> dict:
    """
    Analyzes an already-anonymized image using Qwen2.5-VL.

    Returns:
        description: Human-readable scene description.
        traffic_density: Estimated traffic density.
        limitations: List of analysis limitations.
        report: Complete structured VLM report.
    """

    if not image_bytes:
        raise ValueError("Anonymized image data is missing.")

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    context = f"""
    User's analysis focus:
    {user_prompt}

    YOLOP vehicle detection counts:
    {json.dumps(detection_summary)}

    Privacy preprocessing:
    Faces blurred: {faces_blurred}
    License plates blurred: {plates_blurred}

    Analyze the supplied anonymized image and return the
    structured JSON report requested in the system instructions.
    """

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": context,
                "images": [image_base64]
            }
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))

        content = result["message"]["content"]
        report = json.loads(content)

        return {
            "description": report.get("scene_description", ""),
            "traffic_density": report.get(
                "traffic_observations", {}
            ).get("traffic_density", "Unknown"),
            "limitations": report.get("limitations", []),
            "report": report
        }

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not connect to Ollama: {error}"
        ) from error

    except (KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Invalid response from Qwen: {error}"
        ) from error
