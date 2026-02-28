"""
Tracking Layer - ByteTrack + Kalman Filter
Задача: каждому объекту дать ID и отслеживать траекторию
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import time
from filterpy.kalman import KalmanFilter

from detector import Detection

@dataclass
class Track:
    """Класс для хранения трека"""
    id: int
    class_name: str
    center: Tuple[float, float]
    velocity: Tuple[float, float]
    acceleration: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    confidence: float
    last_seen: float
    trajectory: deque  # последние N позиций
    disappeared_count: int
    kalman_filter: KalmanFilter

class Tracker:
    """Трекер объектов на основе ByteTrack + Kalman Filter"""
    
    def __init__(self, max_disappeared: int = 30, trajectory_length: int = 20):
        self.tracks = {}  # {track_id: Track}
        self.next_track_id = 1
        self.max_disappeared = max_disappeared
        self.trajectory_length = trajectory_length
        
        # Параметры ассоциации
        self.max_distance = 100  # максимальное расстояние для ассоциации
        self.max_iou_distance = 0.7  # максимальное IoU расстояние
        
    def create_kalman_filter(self, initial_center: Tuple[float, float]) -> KalmanFilter:
        """Создание Kalman фильтра для нового трека"""
        kf = KalmanFilter(dim_x=6, dim_z=2)  # [x, y, vx, vy, ax, ay] и измерения [x, y]
        
        # Матрица перехода состояния
        dt = 0.1  # 10 FPS
        kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Матрица измерений
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        
        # Ковариации
        kf.P *= 1000
        kf.R *= 10
        kf.Q *= 0.1
        
        # Начальное состояние
        kf.x = np.array([initial_center[0], initial_center[1], 0, 0, 0, 0]).reshape(6, 1)
        
        return kf
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """Расчет IoU между двумя bbox"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Площадь пересечения
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Площади bbox
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # IoU
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def associate_detections_to_tracks(self, detections: List[Detection]) -> Dict[int, Detection]:
        """Ассоциация детекций с существующими треками"""
        if not detections:
            return {}
        
        unmatched_detections = list(range(len(detections)))
        matches = {}
        
        # Сначала пытаемся сопоставить по IoU
        for track_id, track in self.tracks.items():
            if track.disappeared_count > 0:
                continue
            
            best_detection_idx = None
            best_iou = 0
            
            for det_idx in unmatched_detections:
                detection = detections[det_idx]
                
                # Проверяем класс
                if detection.class_name != track.class_name:
                    continue
                
                # Расчет IoU
                iou = self.calculate_iou(detection.bbox, track.bbox)
                
                if iou > best_iou and iou > self.max_iou_distance:
                    best_iou = iou
                    best_detection_idx = det_idx
            
            if best_detection_idx is not None:
                matches[track_id] = detections[best_detection_idx]
                unmatched_detections.remove(best_detection_idx)
        
        # Для оставшихся детекций пробуем ассоциацию по расстоянию
        for track_id, track in self.tracks.items():
            if track.disappeared_count > 0 or track_id in matches:
                continue
            
            best_detection_idx = None
            best_distance = float('inf')
            
            for det_idx in unmatched_detections:
                detection = detections[det_idx]
                
                if detection.class_name != track.class_name:
                    continue
                
                # Расчет расстояния между центрами
                distance = np.linalg.norm(
                    np.array(detection.center) - np.array(track.center)
                )
                
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_detection_idx = det_idx
            
            if best_detection_idx is not None:
                matches[track_id] = detections[best_detection_idx]
                unmatched_detections.remove(best_detection_idx)
        
        return matches
    
    def update_tracks(self, detections: List[Detection]) -> List[Track]:
        """Обновление треков"""
        current_time = time.time()
        
        # Ассоциация детекций с треками
        matches = self.associate_detections_to_tracks(detections)
        
        # Обновляем сопоставленные треки
        for track_id, detection in matches.items():
            track = self.tracks[track_id]
            
            # Предсказание Kalman
            track.kalman_filter.predict()
            
            # Обновление измерением
            track.kalman_filter.update(np.array(detection.center))
            
            # Получаем состояние из Kalman
            state = track.kalman_filter.x.flatten()
            
            # Обновляем трек
            prev_center = track.center
            track.center = (state[0], state[1])
            track.velocity = (state[2], state[3])
            track.acceleration = (state[4], state[5])
            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.last_seen = current_time
            track.disappeared_count = 0
            
            # Добавляем в траекторию
            track.trajectory.append({
                'center': track.center,
                'velocity': track.velocity,
                'timestamp': current_time
            })
            
            # Ограничиваем длину траектории
            if len(track.trajectory) > self.trajectory_length:
                track.trajectory.popleft()
        
        # Увеличиваем счетчик исчезновения для несопоставленных треков
        for track_id in self.tracks:
            if track_id not in matches:
                self.tracks[track_id].disappeared_count += 1
                # Предсказание для исчезнувших треков
                self.tracks[track_id].kalman_filter.predict()
                
                # Обновляем состояние из предсказания
                state = self.tracks[track_id].kalman_filter.x.flatten()
                self.tracks[track_id].center = (state[0], state[1])
                self.tracks[track_id].velocity = (state[2], state[3])
                self.tracks[track_id].acceleration = (state[4], state[5])
        
        # Создаем новые треки для несопоставленных детекций
        for detection in detections:
            if detection not in matches.values():
                # Создаем новый трек
                kf = self.create_kalman_filter(detection.center)
                
                new_track = Track(
                    id=self.next_track_id,
                    class_name=detection.class_name,
                    center=detection.center,
                    velocity=(0.0, 0.0),
                    acceleration=(0.0, 0.0),
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    last_seen=current_time,
                    trajectory=deque([{
                        'center': detection.center,
                        'velocity': (0.0, 0.0),
                        'timestamp': current_time
                    }], maxlen=self.trajectory_length),
                    disappeared_count=0,
                    kalman_filter=kf
                )
                
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        # Удаляем старые треки
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.disappeared_count > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Возвращаем активные треки
        active_tracks = [track for track in self.tracks.values() 
                        if track.disappeared_count == 0]
        
        return active_tracks
    
    def get_tracks_by_class(self, class_names: List[str]) -> List[Track]:
        """Получение треков по классам"""
        return [track for track in self.tracks.values() 
                if track.class_name in class_names and track.disappeared_count == 0]
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Отрисовка треков на кадре"""
        vis_frame = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            
            # Цвет по классу
            color = self.get_class_color(track.class_name)
            
            # Рисуем bbox с ID
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, f"ID:{track.id}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Рисуем траекторию
            if len(track.trajectory) > 1:
                points = [point['center'] for point in track.trajectory]
                for i in range(len(points) - 1):
                    pt1 = (int(points[i][0]), int(points[i][1]))
                    pt2 = (int(points[i+1][0]), int(points[i+1][1]))
                    cv2.line(vis_frame, pt1, pt2, color, 2)
            
            # Рисуем вектор скорости
            if track.velocity != (0.0, 0.0):
                center = (int(track.center[0]), int(track.center[1]))
                end_point = (
                    int(track.center[0] + track.velocity[0] * 10),
                    int(track.center[1] + track.velocity[1] * 10)
                )
                cv2.arrowedLine(vis_frame, center, end_point, color, 2)
        
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
    
    def get_statistics(self) -> Dict:
        """Получение статистики трекинга"""
        active_tracks = [t for t in self.tracks.values() if t.disappeared_count == 0]
        
        class_distribution = defaultdict(int)
        for track in active_tracks:
            class_distribution[track.class_name] += 1
        
        return {
            'total_tracks': len(self.tracks),
            'active_tracks': len(active_tracks),
            'class_distribution': dict(class_distribution),
            'next_track_id': self.next_track_id
        }

if __name__ == "__main__":
    # Тестирование трекера
    from detector import Detector
    
    detector = Detector()
    tracker = Tracker()
    
    cap = cv2.VideoCapture(0)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Детекция
        detections = detector.detect(frame)
        
        # Трекинг
        tracks = tracker.update_tracks(detections)
        
        # Визуализация
        vis_frame = tracker.draw_tracks(frame, tracks)
        
        # Информация
        cv2.putText(vis_frame, f"Active tracks: {len(tracks)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        stats = tracker.get_statistics()
        cv2.putText(vis_frame, f"Classes: {stats['class_distribution']}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Tracking Layer', vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
