import cv2
import importlib
import os


def get_function(target: str):
    module_path, function_name = target.rsplit('.', 1)

    module = importlib.import_module(module_path)
    function = getattr(module, function_name)

    return function


def save_as_mp4(frames: list, path: str, fps: int = 20, to_bgr: bool = True):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if frames[0].shape[0] == 3:
        frames = [frame.transpose(1, 2, 0) for frame in frames]
    
    shape = frames[0].shape
    out = cv2.VideoWriter(path, fourcc, fps, (shape[1], shape[0]))
    for frame in frames:
        if to_bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
