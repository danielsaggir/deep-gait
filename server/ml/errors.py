"""Domain errors for the production ML runtime."""

from __future__ import annotations


class DeepGaitMLError(Exception):
    code = "ML_ERROR"

    def __init__(self, message: str, *, subject: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.subject = subject

    def to_dict(self) -> dict:
        payload = {"code": self.code, "message": self.message}
        if self.subject is not None:
            payload["subject"] = self.subject
        return payload


class InsufficientGaitDataError(DeepGaitMLError):
    code = "INSUFFICIENT_GAIT_DATA"


class CheckpointError(DeepGaitMLError):
    code = "CHECKPOINT_ERROR"


class VideoDecodeError(DeepGaitMLError):
    code = "VIDEO_DECODE_ERROR"
