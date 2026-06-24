from fastapi import Request


def get_pipeline(request: Request):
    return request.app.state.pipeline


def get_telemetry(request: Request):
    return request.app.state.telemetry
