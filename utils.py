"""
utils.py

Утилиты для отрисовки детекций на кадрах видео.
"""

import cv2
from typing import List, Tuple
import numpy as np


def draw_boxes_on_frame(
    frame: np.ndarray, 
    detections: List[Tuple[int, int, int, int, float]]
) -> np.ndarray:
    """
    Рисует рамки и подписи для каждого найденного человека.

    Args:
        frame (np.ndarray): Кадр OpenCV в формате BGR.
        detections (List[Tuple[int, int, int, int, float]]): 
            Список кортежей с координатами рамки и уверенностью 
            (x1, y1, x2, y2, confidence).

    Returns:
        np.ndarray: Изменённый кадр с отрисованными рамками и подписями.
    """
    for (x1, y1, x2, y2, conf) in detections:
        label = f"person {int(conf * 100)}%"
        # Рисуем зелёный прямоугольник (рамку)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Получаем размер текста, чтобы нарисовать фон под надписью
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

        # Рисуем фон для текста
        cv2.rectangle(frame, (x1, label_y - label_size[1]), (x1 + label_size[0], label_y), (0, 255, 0), cv2.FILLED)

        # Рисуем текст (чёрным цветом)
        cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame
