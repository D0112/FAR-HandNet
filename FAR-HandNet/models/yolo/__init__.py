# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo import classify, detect, pose, segment, pose_scconv,detect_scconv

from .model import YOLO

__all__ = 'classify', 'segment', 'detect', 'pose', 'pose_scconv', 'detect_scconv', 'YOLO'
