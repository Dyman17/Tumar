"""
Detection Layer - YOLOv8 Object Detection
Вход: кадр
Выход: список детекций с bbox и confidence
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import time
from dataclasses import dataclass

@dataclass
class Detection:
    """Класс для хранения детекции"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]

class Detector:
    """Детектор объектов на основе YOLOv8"""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Интересующие классы COCO
        self.target_classes = {
            0: 'person',
            2: 'car', 
            5: 'bus',
            7: 'truck'
        }
        
        # Статистика
        self.processing_times = []
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Детекция объектов на кадре"""
        start_time = time.time()
        
        # YOLO inference
        results = self.model(frame, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Фильтруем только нужные классы
                    if class_id in self.target_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detection = Detection(
                            class_name=self.target_classes[class_id],
                            confidence=confidence,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            center=(int((x1 + x2) / 2), int((y1 + y2) / 2))
                        )
                        
                        detections.append(detection)
        
        # Сохраняем время обработки
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return detections
    
    def get_average_processing_time(self) -> float:
        """Среднее время обработки"""
        if not self.processing_times:
            return 0.0
        return np.mean(self.processing_times[-100:])  # Последние 100 кадров
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Отрисовка детекций на кадре"""
        vis_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Цвет по классу
            color = self.get_class_color(detection.class_name)
            
            # Рисуем bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Текст с классом и уверенностью
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            
            # Размер текста
            (text_width, text_height) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Фон для текста
            cv2.rectangle(vis_frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Текст
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_frame
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Получение цвета для класса"""
        colors = {
            'person': (0, 255, 0),      # Зеленый
            'car': (255, 0, 0),         # Красный
            'truck': (0, 0, 255),       # Синий
            'bus': (255, 255, 0)        # Желтый
        }
        return colors.get(class_name, (255, 255, 255))

if __name__ == "__main__":
    # Тестирование детектора
    detector = Detector()
    
    # Тест с веб-камерой
    cap = cv2.VideoCapture(0)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        vis_frame = detector.draw_detections(frame, detections)
        
        # Информация
        cv2.putText(vis_frame, f"Detections: {len(detections)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Avg time: {detector.get_average_processing_time():.3f}s", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Detection Layer', vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
