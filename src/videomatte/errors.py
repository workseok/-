"""사용자에게 그대로 보여줄 수 있는 명확한 에러 메시지용 예외."""


class VideomatteError(Exception):
    """CLI 가 traceback 없이 메시지만 출력하는 예외의 베이스."""


class FFmpegNotFoundError(VideomatteError):
    pass


class MissingModelError(VideomatteError):
    pass


class DetectionError(VideomatteError):
    pass


class OutOfMemoryHint(VideomatteError):
    pass


class ConfigError(VideomatteError):
    pass
