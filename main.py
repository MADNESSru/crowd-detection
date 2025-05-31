import os
import cv2

from detector import Detector
from utils import draw_boxes_on_frame


def main() -> None:
    """
    Основная функция программы.

    Загружает видеофайл 'crowd.mp4', выполняет детекцию людей с помощью модели YOLOv8,
    отрисовывает рамки и подписи на кадрах и сохраняет результат в 'output/result_final.mp4'.
    """
    input_path = 'crowd.mp4'
    output_path = os.path.join('output', 'result_final.mp4')

    os.makedirs('output', exist_ok=True)

    detector = Detector(model_path='yolov8l.pt', upscale_factor=1.0)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видеофайл {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Ошибка: не удалось создать файл для записи видео {output_path}")
        cap.release()
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        frame = draw_boxes_on_frame(frame, detections)

        out.write(frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Обработано кадров: {frame_count}")

    cap.release()
    out.release()
    print(f"Готово. Видео сохранено в: {output_path}")


if __name__ == "__main__":
    main()
