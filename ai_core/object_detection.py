import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass
from collections import defaultdict, deque
import json

@dataclass
class Detection:
    """Класс для хранения детекции"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]
    track_id: Optional[int] = None
    timestamp: float = None

class ObjectDetectionEngine:
    """Основной движок детекции объектов"""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Классы COCO, которые нас интересуют
        self.target_classes = {
            0: 'person',
            1: 'bicycle', 
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            15: 'cat',
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow',
            20: 'elephant',
            21: 'bear',
            22: 'zebra',
            23: 'giraffe',
            27: 'fire'  # Если есть кастомная тренировка
        }
        
        # Статистика производительности
        self.processing_times = deque(maxlen=100)
        self.detection_stats = defaultdict(int)
        
    def detect_objects(self, frame: np.ndarray, frame_id: int = None) -> List[Detection]:
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
                            center=(int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            timestamp=time.time()
                        )
                        
                        detections.append(detection)
                        self.detection_stats[detection.class_name] += 1
        
        # Сохраняем время обработки
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return detections
    
    def get_detection_summary(self, detections: List[Detection]) -> Dict:
        """Получение сводки детекций"""
        summary = defaultdict(int)
        for detection in detections:
            summary[detection.class_name] += 1
        
        return dict(summary)
    
    def get_performance_stats(self) -> Dict:
        """Статистика производительности"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'fps_estimate': 1.0 / np.mean(self.processing_times) if np.mean(self.processing_times) > 0 else 0,
            'total_detections': dict(self.detection_stats)
        }
    
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
            if detection.track_id is not None:
                label += f" ID:{detection.track_id}"
            
            # Размер текста
            (text_width, text_height) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Фон для текста
            cv2.rectangle(vis_frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Текст
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Центральная точка
            cv2.circle(vis_frame, detection.center, 3, color, -1)
        
        return vis_frame
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Получение цвета для класса"""
        colors = {
            'person': (0, 255, 0),      # Зеленый
            'car': (255, 0, 0),         # Красный
            'truck': (0, 0, 255),       # Синий
            'motorcycle': (255, 255, 0), # Желтый
            'bicycle': (255, 0, 255),   # Пурпурный
            'bus': (0, 255, 255),       # Бирюзовый
            'fire': (0, 69, 255),       # Оранжевый
            'animal': (128, 128, 128)   # Серый
        }
        return colors.get(class_name, (255, 255, 255))
    
    def export_detections_json(self, detections: List[Detection], frame_id: int) -> str:
        """Экспорт детекций в JSON"""
        data = {
            'frame_id': frame_id,
            'timestamp': time.time(),
            'detections': [
                {
                    'class': det.class_name,
                    'confidence': det.confidence,
                    'bbox': det.bbox,
                    'center': det.center,
                    'track_id': det.track_id
                }
                for det in detections
            ]
        }
        return json.dumps(data, indent=2)

class MultiObjectTracker:
    """Мультиобъектный трекер"""
    
    def __init__(self, max_disappeared: int = 30):
        self.tracks = {}  # {track_id: Track}
        self.next_track_id = 1
        self.max_disappeared = max_disappeared
        
    def update_tracks(self, detections: List[Detection]) -> List[Detection]:
        """Обновление треков и присвоение ID"""
        if not detections:
            # Увеличиваем счетчик исчезновения для всех треков
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return []
        
        # Простая ассоциация по ближайшему центру
        unmatched_detections = detections.copy()
        matched_detections = []
        
        # Пытаемся сопоставить существующие треки
        for track_id, track in self.tracks.items():
            if track['disappeared'] > 0:
                continue
                
            best_detection = None
            best_distance = float('inf')
            
            for detection in unmatched_detections:
                distance = np.linalg.norm(
                    np.array(detection.center) - np.array(track['last_center'])
                )
                
                if distance < 100 and distance < best_distance:  # Порог 100 пикселей
                    best_distance = distance
                    best_detection = detection
            
            if best_detection:
                best_detection.track_id = track_id
                track['last_center'] = best_detection.center
                track['disappeared'] = 0
                track['last_seen'] = time.time()
                
                matched_detections.append(best_detection)
                unmatched_detections.remove(best_detection)
            else:
                track['disappeared'] += 1
        
        # Создаем новые треки
        for detection in unmatched_detections:
            detection.track_id = self.next_track_id
            
            self.tracks[self.next_track_id] = {
                'last_center': detection.center,
                'disappeared': 0,
                'last_seen': time.time(),
                'class': detection.class_name
            }
            
            self.next_track_id += 1
            matched_detections.append(detection)
        
        # Удаляем старые треки
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                del self.tracks[track_id]
        
        return matched_detections
    
    def get_track_statistics(self) -> Dict:
        """Статистика треков"""
        active_tracks = len([t for t in self.tracks.values() if t['disappeared'] == 0])
        class_distribution = defaultdict(int)
        
        for track in self.tracks.values():
            if track['disappeared'] == 0:
                class_distribution[track['class']] += 1
        
        return {
            'active_tracks': active_tracks,
            'total_tracks': len(self.tracks),
            'class_distribution': dict(class_distribution)
        }

# Интеграция детекции и трекинга
class DetectionSystem:
    """Полная система детекции и трекинга"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.detector = ObjectDetectionEngine(model_path)
        self.tracker = MultiObjectTracker()
        self.frame_count = 0
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Полная обработка кадра"""
        self.frame_count += 1
        
        # Детекция
        detections = self.detector.detect_objects(frame, self.frame_count)
        
        # Трекинг
        tracked_detections = self.tracker.update_tracks(detections)
        
        # Визуализация
        vis_frame = self.detector.draw_detections(frame, tracked_detections)
        
        # Статистика
        detection_summary = self.detector.get_detection_summary(tracked_detections)
        track_stats = self.tracker.get_track_statistics()
        performance_stats = self.detector.get_performance_stats()
        
        return {
            'frame_id': self.frame_count,
            'detections': tracked_detections,
            'detection_summary': detection_summary,
            'track_stats': track_stats,
            'performance_stats': performance_stats,
            'vis_frame': vis_frame,
            'timestamp': time.time()
        }

if __name__ == "__main__":
    # Тестирование системы
    detection_system = DetectionSystem()
    
    # Тест с веб-камерой
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = detection_system.process_frame(frame)
        
        # Отображение результатов
        cv2.imshow('Detection System', results['vis_frame'])
        
        # Печать статистики каждые 30 кадров
        if results['frame_id'] % 30 == 0:
            print(f"Frame {results['frame_id']}: {results['detection_summary']}")
            print(f"Active tracks: {results['track_stats']['active_tracks']}")
            print(f"Avg processing time: {results['performance_stats'].get('avg_processing_time', 0):.3f}s")
            print("-" * 50)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
