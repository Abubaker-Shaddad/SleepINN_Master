"""Independent GLM request adapter for the frozen clinical rubric."""
from . import clinical_judge as base

MODEL = "z-ai/glm-5.3-flash"


def request_for(job, protocol, rubric):
    text = base.make_request(job, {"generationConfig": {}}, rubric)["contents"][0][
        "parts"
    ][0]["text"]
    return {
        "model": protocol["judge_model"],
        "messages": [{"role": "user", "content": text}],
        **protocol["request_settings"],
    }


def validate(raw, job, protocol):
    if raw.get("model") not in {MODEL, protocol["catalog_canonical_slug"]}:
        raise ValueError("Returned model differs from the protocol")
    if raw.get("provider") != protocol["provider_name"]:
        raise ValueError("Returned provider differs from the protocol")
    choices = raw.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Incomplete or refused judge response")
    message = choices[0].get("message", {})
    if message.get("refusal") or not isinstance(message.get("content"), str):
        raise ValueError("Missing judge JSON")
    adapted = {
        "modelVersion": base.MODEL,
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"text": message["content"]}]},
            }
        ],
    }
    return base.validate(adapted, job)
