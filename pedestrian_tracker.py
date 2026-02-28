import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from filterpy.kalman import KalmanFilter
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import asyncio
import websockets
import threading

class PedestrianTracker:
    def __init__(self, model_path="yolov8n.pt"):
        # Инициализация YOLO модели
        self.model = YOLO(model_path)
        
        # Настройки детекции
        self.confidence_threshold = 0.5
        self.target_class = 0  # 'person' class в COCO
        
        # Трекинг
        self.tracks = {}  # {track_id: Track}
        self.next_track_id = 1
        self.max_disappeared = 30  # кадры без детекции перед удалением
        
        # Предсказание траекторий
        self.prediction_horizon = 10  # кадров вперед
        
        # Буфер для анализа плотности
        self.density_buffer = deque(maxlen=100)
        
        # История траекторий
        self.trajectory_history = defaultdict(lambda: deque(maxlen=50))
        
    def detect_pedestrians(self, frame: np.ndarray) -> List[Dict]:
        """Детекция пешеходов на кадре"""
        results = self.model(frame, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) == self.target_class:  # только люди
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                        })
        
        return detections
    
    def update_tracks(self, detections: List[Dict], frame_idx: int):
        """Обновление треков с простым ассоциацией по IoU"""
        if not detections:
            # Увеличиваем счетчик исчезновения для всех треков
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return
        
        # Простой трекинг на основе ближайшего центра
        unmatched_detections = list(range(len(detections)))
        
        # Пытаемся сопоставить существующие треки
        for track_id, track in self.tracks.items():
            if track['disappeared'] > 0:
                continue
                
            best_det_idx = None
            best_distance = float('inf')
            
            for det_idx in unmatched_detections:
                det_center = detections[det_idx]['center']
                track_center = track['kalman'].x[:2].flatten()
                
                distance = np.linalg.norm(det_center - track_center)
                
                if distance < 100 and distance < best_distance:  # порог 100 пикселей
                    best_distance = distance
                    best_det_idx = det_idx
            
            if best_det_idx is not None:
                # Обновляем трек
                detection = detections[best_det_idx]
                track['bbox'] = detection['bbox']
                track['disappeared'] = 0
                track['last_seen'] = frame_idx
                
                # Обновляем Kalman фильтр
                track['kalman'].update(detection['center'])
                
                # Сохраняем в историю
                self.trajectory_history[track_id].append({
                    'frame': frame_idx,
                    'center': detection['center'],
                    'bbox': detection['bbox'],
                    'timestamp': time.time()
                })
                
                unmatched_detections.remove(best_det_idx)
            else:
                track['disappeared'] += 1
        
        # Создаем новые треки для несовпадающих детекций
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            
            # Создаем Kalman фильтр для нового трека
            kf = self.create_kalman_filter()
            kf.x[:2] = np.array(detection['center']).reshape(2, 1)
            
            self.tracks[self.next_track_id] = {
                'id': self.next_track_id,
                'bbox': detection['bbox'],
                'kalman': kf,
                'disappeared': 0,
                'last_seen': frame_idx,
                'created': frame_idx
            }
            
            self.trajectory_history[self.next_track_id].append({
                'frame': frame_idx,
                'center': detection['center'],
                'bbox': detection['bbox'],
                'timestamp': time.time()
            })
            
            self.next_track_id += 1
        
        # Удаляем старые треки
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                del self.tracks[track_id]
    
    def create_kalman_filter(self):
        """Создание Kalman фильтра для трекинга позиции"""
        kf = KalmanFilter(dim_x=4, dim_z=2)  # [x, y, vx, vy] и измерения [x, y]
        
        # Матрица перехода состояния
        kf.F = np.array([[1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        
        # Матрица измерений
        kf.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
        
        # Ковариация процесса и измерений
        kf.P *= 1000
        kf.R *= 10
        kf.Q *= 0.1
        
        return kf
    
    def predict_trajectories(self) -> Dict[int, List[Tuple[int, int]]]:
        """Предсказание траекторий для всех активных треков"""
        predictions = {}
        
        for track_id, track in self.tracks.items():
            if track['disappeared'] > 0:
                continue
                
            # Копируем фильтр для предсказания
            kf = track['kalman']
            
            trajectory = []
            x_pred = kf.x.copy()
            
            for _ in range(self.prediction_horizon):
                x_pred = kf.F @ x_pred
                trajectory.append((int(x_pred[0]), int(x_pred[1])))
            
            predictions[track_id] = trajectory
        
        return predictions
    
    def calculate_density(self) -> Dict:
        """Расчет плотности пешеходов"""
        active_tracks = len([t for t in self.tracks.values() if t['disappeared'] == 0])
        
        # Сохраняем в буфер
        self.density_buffer.append({
            'timestamp': time.time(),
            'count': active_tracks
        })
        
        # Средняя плотность за последние 10 секунд
        recent_density = np.mean([d['count'] for d in self.density_buffer if 
                                time.time() - d['timestamp'] < 10])
        
        return {
            'current': active_tracks,
            'average_10s': float(recent_density),
            'trend': 'increasing' if active_tracks > recent_density else 'decreasing'
        }
    
    def get_heatmap_data(self) -> List[Dict]:
        """Получение данных для heatmap"""
        heatmap_points = []
        
        for track_id, history in self.trajectory_history.items():
            for point in history:
                heatmap_points.append({
                    'x': point['center'][0],
                    'y': point['center'][1],
                    'timestamp': point['timestamp']
                })
        
        return heatmap_points
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict:
        """Обработка одного кадра"""
        # Детекция
        detections = self.detect_pedestrians(frame)
        
        # Трекинг
        self.update_tracks(detections, frame_idx)
        
        # Предсказание
        predictions = self.predict_trajectories()
        
        # Плотность
        density = self.calculate_density()
        
        return {
            'frame_idx': frame_idx,
            'detections': detections,
            'tracks': self.tracks,
            'predictions': predictions,
            'density': density,
            'timestamp': time.time()
        }
    
    def draw_visualization(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Отрисовка визуализации на кадре"""
        vis_frame = frame.copy()
        
        # Рисуем детекции
        for det in results['detections']:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Рисуем треки и предсказания
        for track_id, track in results['tracks'].items():
            if track['disappeared'] > 0:
                continue
                
            x1, y1, x2, y2 = track['bbox']
            
            # Цвет для трека
            color = (0, 0, 255) if track['disappeared'] == 0 else (128, 128, 128)
            
            # Рисуем bbox с ID
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, f"ID:{track_id}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Рисуем предсказанную траекторию
            if track_id in results['predictions']:
                pred_points = results['predictions'][track_id]
                for i in range(len(pred_points) - 1):
                    pt1 = pred_points[i]
                    pt2 = pred_points[i + 1]
                    alpha = 1.0 - (i / len(pred_points))  # затухание
                    color_pred = (255, int(255 * alpha), int(255 * alpha))
                    cv2.line(vis_frame, pt1, pt2, color_pred, 2)
        
        # Информация о плотности
        density = results['density']
        cv2.putText(vis_frame, f"People: {density['current']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Avg: {density['average_10s']:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame

if __name__ == "__main__":
    # Тестовый запуск
    tracker = PedestrianTracker()
    
    # Для теста с веб-камерой
    cap = cv2.VideoCapture(0)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = tracker.process_frame(frame, frame_idx)
        vis_frame = tracker.draw_visualization(frame, results)
        
        cv2.imshow('Pedestrian Tracking', vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
