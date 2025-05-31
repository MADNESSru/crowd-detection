"""
detector.py

Модуль с классом Detector для детекции людей на изображениях с использованием YOLOv8.
"""

from ultralytics import YOLO
import cv2
from typing import List, Tuple


class Detector:
    """
    Класс-обёртка над моделью YOLOv8 для детекции людей на изображениях.

    Attributes:
        model (YOLO): Загрузка модели YOLOv8.
        upscale_factor (float): Масштаб увеличения изображения перед детекцией.
    """

    def __init__(self, model_path: str = 'yolov8l.pt', upscale_factor: float = 1.0) -> None:
        """
        Инициализирует детектор с заданной моделью и масштабом апскейла.

        Args:
            model_path (str): Путь к весам модели YOLOv8.
            upscale_factor (float): Масштаб увеличения изображения перед детекцией.
        """
        self.model = YOLO(model_path)
        self.upscale_factor = upscale_factor

    def detect(self, frame: 'numpy.ndarray') -> List[Tuple[int, int, int, int, float]]:
        """
        Выполняет детекцию людей (class_id == 0) на кадре.

        Args:
            frame (numpy.ndarray): Кадр OpenCV (формат BGR).

        Returns:
            List[Tuple[int, int, int, int, float]]: Список кортежей (x1, y1, x2, y2, confidence) для найденных людей.
        """
        if self.upscale_factor != 1.0:
            frame = cv2.resize(frame, None, fx=self.upscale_factor, fy=self.upscale_factor)

        results = self.model(frame)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if self.upscale_factor != 1.0:
                    scale = 1 / self.upscale_factor
                    x1, y1, x2, y2 = map(lambda v: int(v * scale), (x1, y1, x2, y2))
                detections.append((x1, y1, x2, y2, conf))

        return detections
