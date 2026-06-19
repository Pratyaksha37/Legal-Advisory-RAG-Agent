from src.config.settings import settings


class ConfidenceThreshold:
    def __init__(self, threshold: float | None = None):
        self.threshold = threshold if threshold is not None else settings.confidence_threshold

    def check(self, confidence_score: float) -> tuple[bool, str]:
        if confidence_score >= self.threshold:
            return True, ""
        return False, f"Confidence score {confidence_score:.3f} below threshold {self.threshold}"
