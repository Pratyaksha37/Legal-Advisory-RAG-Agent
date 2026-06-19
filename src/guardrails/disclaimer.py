from src.config.constants import DEFAULT_DISCLAIMER


class Disclaimer:
    def __init__(self, text: str | None = None):
        self.text = text or DEFAULT_DISCLAIMER

    def append(self, response: str) -> str:
        if self.text in response:
            return response
        return response.rstrip() + "\n\n" + self.text
