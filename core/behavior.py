"""
Behavior Analysis Layer - Анализ поведения объектов
A. Группировка техники (convoy detection)
B. Аномальная скорость
C. Резкое изменение направления
D. Плотность объектов
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import math

from tracker import Track

@dataclass
class BehaviorFeatures:
    """Класс для хранения поведенческих признаков"""
    convoy_flag: bool
    convoy_size: int
    anomaly_speed: bool
    anomaly_direction: bool
    density: float
    grid_density: Dict[str, float]
    avg_scene_speed: float
    speed_variance: float

class BehaviorAnalyzer:
    """Анализатор поведения объектов"""
    
    def __init__(self):
        # Параметры анализа
        self.convoy_distance_threshold = 150  # пикселей
        self.convoy_size_threshold = 3  # минимальный размер конвоя
        self.speed_anomaly_factor = 2.0  # множитель для аномальной скорости
        self.direction_change_threshold = 45  # градусов
        self.grid_size = 4  # сетка 4x4
        
        # Исторические данные
        self.scene_speed_history = []
        self.max_history_length = 100
        
    def analyze(self, tracks: List[Track]) -> BehaviorFeatures:
        """Комплексный анализ поведения"""
        
        # A. Группировка техники
        convoy_flag, convoy_size = self.detect_convoy(tracks)
        
        # B. Аномальная скорость
        anomaly_speed = self.detect_speed_anomaly(tracks)
        
        # C. Резкое изменение направления
        anomaly_direction = self.detect_direction_anomaly(tracks)
        
        # D. Плотность
        density, grid_density = self.calculate_density(tracks)
        
        # Дополнительные метрики
        avg_scene_speed, speed_variance = self.calculate_scene_speed_stats(tracks)
        
        return BehaviorFeatures(
            convoy_flag=convoy_flag,
            convoy_size=convoy_size,
            anomaly_speed=anomaly_speed,
            anomaly_direction=anomaly_direction,
            density=density,
            grid_density=grid_density,
            avg_scene_speed=avg_scene_speed,
            speed_variance=speed_variance
        )
    
    def detect_convoy(self, tracks: List[Track]) -> Tuple[bool, int]:
        """A. Группировка техники (convoy detection)"""
        
        # Фильтруем только технику
        vehicle_tracks = [track for track in tracks 
                         if track.class_name in ['car', 'truck', 'bus']]
        
        if len(vehicle_tracks) < self.convoy_size_threshold:
            return False, 0
        
        # Строим граф связей на основе расстояния и схожести движения
        convoy_groups = []
        used_tracks = set()
        
        for i, track1 in enumerate(vehicle_tracks):
            if track1 in used_tracks:
                continue
            
            # Начинаем новую группу
            current_group = [track1]
            used_tracks.add(track1)
            
            # Ищем связанные треки
            for j, track2 in enumerate(vehicle_tracks):
                if i == j or track2 in used_tracks:
                    continue
                
                # Проверяем условия для группировки
                if self._are_tracks_in_convoy(track1, track2):
                    current_group.append(track2)
                    used_tracks.add(track2)
            
            convoy_groups.append(current_group)
        
        # Находим самую большую группу
        if convoy_groups:
            largest_group = max(convoy_groups, key=len)
            convoy_size = len(largest_group)
            convoy_flag = convoy_size >= self.convoy_size_threshold
            
            return convoy_flag, convoy_size
        
        return False, 0
    
    def _are_tracks_in_convoy(self, track1: Track, track2: Track) -> bool:
        """Проверка, находятся ли два трека в одном конвое"""
        
        # 1. Расстояние между центрами
        distance = np.linalg.norm(
            np.array(track1.center) - np.array(track2.center)
        )
        
        if distance > self.convoy_distance_threshold:
            return False
        
        # 2. Направление движения должно быть похожим
        if track1.velocity != (0.0, 0.0) and track2.velocity != (0.0, 0.0):
            # Вычисляем угол между векторами скорости
            angle = self._calculate_angle_between_vectors(
                track1.velocity, track2.velocity
            )
            
            if angle > 30:  # градусов
                return False
        
        # 3. Скорость должна быть похожей
        speed1 = np.linalg.norm(track1.velocity)
        speed2 = np.linalg.norm(track2.velocity)
        
        if speed1 > 0 and speed2 > 0:
            speed_ratio = max(speed1, speed2) / min(speed1, speed2)
            if speed_ratio > 2.0:  # одна скорость больше другой в 2 раза
                return False
        
        return True
    
    def _calculate_angle_between_vectors(self, v1: Tuple[float, float], 
                                       v2: Tuple[float, float]) -> float:
        """Расчет угла между двумя векторами"""
        
        # Нормализация векторов
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        v1_norm = np.array(v1) / norm1
        v2_norm = np.array(v2) / norm2
        
        # Расчет косинуса угла
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def detect_speed_anomaly(self, tracks: List[Track]) -> bool:
        """B. Аномальная скорость"""
        
        if not tracks:
            return False
        
        # Считаем скорости всех объектов
        speeds = []
        for track in tracks:
            speed = np.linalg.norm(track.velocity)
            if speed > 0:  # игнорируем неподвижные объекты
                speeds.append(speed)
        
        if not speeds:
            return False
        
        # Средняя скорость по сцене
        avg_speed = np.mean(speeds)
        
        # Проверяем, есть ли объекты со скоростью > среднего * K
        threshold = avg_speed * self.speed_anomaly_factor
        
        for speed in speeds:
            if speed > threshold:
                return True
        
        return False
    
    def detect_direction_anomaly(self, tracks: List[Track]) -> bool:
        """C. Резкое изменение направления"""
        
        for track in tracks:
            if len(track.trajectory) < 2:
                continue
            
            # Берем последние две точки траектории
            points = list(track.trajectory)
            if len(points) < 2:
                continue
            
            # Вычисляем векторы движения
            current_point = points[-1]
            prev_point = points[-2]
            
            # Вектор текущего движения
            current_vector = (
                current_point['center'][0] - prev_point['center'][0],
                current_point['center'][1] - prev_point['center'][1]
            )
            
            # Если есть предыдущий вектор
            if len(points) >= 3:
                prev_prev_point = points[-3]
                prev_vector = (
                    prev_point['center'][0] - prev_prev_point['center'][0],
                    prev_point['center'][1] - prev_prev_point['center'][1]
                )
                
                # Расчет угла изменения направления
                angle = self._calculate_angle_between_vectors(
                    prev_vector, current_vector
                )
                
                if angle > self.direction_change_threshold:
                    return True
        
        return False
    
    def calculate_density(self, tracks: List[Track]) -> Tuple[float, Dict[str, float]]:
        """D. Плотность объектов"""
        
        if not tracks:
            return 0.0, {}
        
        # Общая плотность (объекты на площадь)
        frame_area = 640 * 480  # предполагаем размер кадра
        total_density = len(tracks) / (frame_area / 1000)  # нормализуем
        
        # Плотность по сетке
        grid_density = self._calculate_grid_density(tracks)
        
        return total_density, grid_density
    
    def _calculate_grid_density(self, tracks: List[Track]) -> Dict[str, float]:
        """Расчет плотности по сетке"""
        
        grid_density = {}
        cell_width = 640 / self.grid_size
        cell_height = 480 / self.grid_size
        
        # Инициализируем сетку
        grid = defaultdict(int)
        
        # Распределяем треки по ячейкам
        for track in tracks:
            x, y = track.center
            grid_x = int(x / cell_width)
            grid_y = int(y / cell_height)
            
            # Ограничиваем индексы
            grid_x = max(0, min(grid_x, self.grid_size - 1))
            grid_y = max(0, min(grid_y, self.grid_size - 1))
            
            grid[f"{grid_x}_{grid_y}"] += 1
        
        # Нормализуем плотность для каждой ячейки
        cell_area = (cell_width * cell_height) / 1000  # нормализуем
        
        for cell_key, count in grid.items():
            grid_density[cell_key] = count / cell_area
        
        return dict(grid_density)
    
    def calculate_scene_speed_stats(self, tracks: List[Track]) -> Tuple[float, float]:
        """Расчет статистики скоростей по сцене"""
        
        speeds = []
        for track in tracks:
            speed = np.linalg.norm(track.velocity)
            speeds.append(speed)
        
        if not speeds:
            return 0.0, 0.0
        
        avg_speed = np.mean(speeds)
        speed_variance = np.var(speeds)
        
        # Сохраняем в историю
        self.scene_speed_history.append(avg_speed)
        if len(self.scene_speed_history) > self.max_history_length:
            self.scene_speed_history.pop(0)
        
        return avg_speed, speed_variance
    
    def draw_behavior_analysis(self, frame: np.ndarray, tracks: List[Track], 
                             features: BehaviorFeatures) -> np.ndarray:
        """Отрисовка анализа поведения"""
        
        vis_frame = frame.copy()
        
        # Отмечаем конвои
        if features.convoy_flag:
            vehicle_tracks = [track for track in tracks 
                            if track.class_name in ['car', 'truck', 'bus']]
            
            # Рисуем связи между треками в конвое
            convoy_groups = self._find_convoy_groups(vehicle_tracks)
            
            for group in convoy_groups:
                if len(group) >= self.convoy_size_threshold:
                    # Рисуем линии между треками в группе
                    for i, track1 in enumerate(group):
                        for track2 in group[i+1:]:
                            cv2.line(vis_frame, 
                                   (int(track1.center[0]), int(track1.center[1])),
                                   (int(track2.center[0]), int(track2.center[1])),
                                   (0, 255, 255), 1)  # Желтый для конвоя
        
        # Отмечаем аномалии скорости
        if features.anomaly_speed:
            for track in tracks:
                speed = np.linalg.norm(track.velocity)
                if speed > features.avg_scene_speed * self.speed_anomaly_factor:
                    # Рисуем красный круг вокруг объекта с аномальной скоростью
                    cv2.circle(vis_frame, 
                             (int(track.center[0]), int(track.center[1])),
                             30, (0, 0, 255), 2)  # Красный
        
        # Отмечаем аномалии направления
        if features.anomaly_direction:
            for track in tracks:
                if len(track.trajectory) >= 3:
                    points = list(track.trajectory)
                    current_vector = (
                        points[-1]['center'][0] - points[-2]['center'][0],
                        points[-1]['center'][1] - points[-2]['center'][1]
                    )
                    prev_vector = (
                        points[-2]['center'][0] - points[-3]['center'][0],
                        points[-2]['center'][1] - points[-3]['center'][1]
                    )
                    
                    angle = self._calculate_angle_between_vectors(
                        prev_vector, current_vector
                    )
                    
                    if angle > self.direction_change_threshold:
                        # Рисуем оранжевый индикатор
                        cv2.circle(vis_frame, 
                                 (int(track.center[0]), int(track.center[1])),
                                 25, (0, 165, 255), 2)  # Оранжевый
        
        # Рисуем сетку плотности
        self._draw_density_grid(vis_frame, features.grid_density)
        
        # Информационная панель
        info_text = [
            f"Convoy: {features.convoy_flag} (size: {features.convoy_size})",
            f"Speed Anomaly: {features.anomaly_speed}",
            f"Direction Anomaly: {features.anomaly_direction}",
            f"Density: {features.density:.3f}",
            f"Avg Speed: {features.avg_scene_speed:.1f}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(vis_frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    
    def _find_convoy_groups(self, tracks: List[Track]) -> List[List[Track]]:
        """Поиск групп конвоя"""
        
        convoy_groups = []
        used_tracks = set()
        
        for i, track1 in enumerate(tracks):
            if track1 in used_tracks:
                continue
            
            current_group = [track1]
            used_tracks.add(track1)
            
            for track2 in tracks:
                if track2 in used_tracks:
                    continue
                
                if self._are_tracks_in_convoy(track1, track2):
                    current_group.append(track2)
                    used_tracks.add(track2)
            
            convoy_groups.append(current_group)
        
        return convoy_groups
    
    def _draw_density_grid(self, frame: np.ndarray, grid_density: Dict[str, float]):
        """Отрисовка сетки плотности"""
        
        cell_width = 640 / self.grid_size
        cell_height = 480 / self.grid_size
        
        # Находим максимальную плотность для нормализации
        max_density = max(grid_density.values()) if grid_density else 1.0
        
        for cell_key, density in grid_density.items():
            grid_x, grid_y = map(int, cell_key.split('_'))
            
            # Координаты ячейки
            x1 = int(grid_x * cell_width)
            y1 = int(grid_y * cell_height)
            x2 = int((grid_x + 1) * cell_width)
            y2 = int((grid_y + 1) * cell_height)
            
            # Интенсивность цвета на основе плотности
            intensity = int((density / max_density) * 255)
            color = (0, intensity, 0)  # Зеленый с разной интенсивностью
            
            # Рисуем полупрозрачный прямоугольник
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Рисуем границы ячеек
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

if __name__ == "__main__":
    # Тестирование анализатора поведения
    from tracker import Tracker
    from detector import Detector
    
    detector = Detector()
    tracker = Tracker()
    behavior_analyzer = BehaviorAnalyzer()
    
    cap = cv2.VideoCapture(0)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Детекция и трекинг
        detections = detector.detect(frame)
        tracks = tracker.update_tracks(detections)
        
        # Анализ поведения
        features = behavior_analyzer.analyze(tracks)
        
        # Визуализация
        vis_frame = behavior_analyzer.draw_behavior_analysis(frame, tracks, features)
        
        cv2.imshow('Behavior Analysis Layer', vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
