import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import List, Dict, Tuple
import json

class TrajectoryPredictor(nn.Module):
    """LSTM модель для предсказания траекторий пешеходов"""
    
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=4, sequence_length=10):
        super(TrajectoryPredictor, self).__init__()
        
        self.input_size = input_size  # [x, y, vx, vy]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.sequence_length = sequence_length
        
        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Полносвязные слои для предсказания
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Инициализация весов
        self.init_weights()
    
    def init_weights(self):
        """Инициализация весов"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        """
        x: tensor of shape (batch_size, sequence_length, input_size)
        """
        # LSTM проход
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Берем последний выход
        last_output = lstm_out[:, -1, :]
        
        # Предсказание следующей точки
        prediction = self.fc(last_output)
        
        return prediction

class TrajectoryManager:
    """Менеджер траекторий с предсказанием"""
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)
        self.model = None
        self.trajectory_buffer = {}  # {track_id: deque of trajectories}
        self.max_buffer_size = 50
        self.sequence_length = 10
        self.prediction_horizon = 5  # секунд вперед
        
        # Загрузка или создание модели
        if model_path and torch.cuda.is_available():
            self.load_model(model_path)
        else:
            self.create_model()
    
    def create_model(self):
        """Создание новой модели"""
        self.model = TrajectoryPredictor(
            input_size=4,
            hidden_size=64,
            num_layers=2,
            output_size=4,
            sequence_length=self.sequence_length
        ).to(self.device)
        
        # Предобученные веса (если есть)
        self.model.eval()
    
    def load_model(self, model_path):
        """Загрузка предобученной модели"""
        try:
            self.model = TrajectoryPredictor().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.create_model()
    
    def add_trajectory_point(self, track_id: int, x: float, y: float, timestamp: float):
        """Добавление точки траектории"""
        if track_id not in self.trajectory_buffer:
            self.trajectory_buffer[track_id] = deque(maxlen=self.max_buffer_size)
        
        # Вычисляем скорость (если есть предыдущая точка)
        vx, vy = 0, 0
        if len(self.trajectory_buffer[track_id]) > 0:
            last_point = self.trajectory_buffer[track_id][-1]
            dt = timestamp - last_point['timestamp']
            if dt > 0:
                vx = (x - last_point['x']) / dt
                vy = (y - last_point['timestamp']) / dt
        
        # Добавляем новую точку
        self.trajectory_buffer[track_id].append({
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'timestamp': timestamp
        })
    
    def prepare_sequence(self, trajectory: deque) -> torch.Tensor:
        """Подготовка последовательности для модели"""
        if len(trajectory) < self.sequence_length:
            return None
        
        # Берем последние sequence_length точек
        sequence_points = list(trajectory)[-self.sequence_length:]
        
        # Формируем тензор [x, y, vx, vy]
        sequence_data = []
        for point in sequence_points:
            sequence_data.append([
                point['x'],
                point['y'],
                point['vx'],
                point['vy']
            ])
        
        return torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(0)  # batch_size=1
    
    def predict_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """Предсказание траектории для трека"""
        if track_id not in self.trajectory_buffer:
            return []
        
        trajectory = self.trajectory_buffer[track_id]
        if len(trajectory) < self.sequence_length:
            # Если данных мало, используем простую линейную экстраполяцию
            return self.linear_extrapolation(trajectory)
        
        # Подготовка данных для модели
        sequence = self.prepare_sequence(trajectory)
        if sequence is None:
            return self.linear_extrapolation(trajectory)
        
        try:
            with torch.no_grad():
                sequence = sequence.to(self.device)
                prediction = self.model(sequence)
                
                # Конвертируем в координаты
                pred_x, pred_y, pred_vx, pred_vy = prediction.cpu().numpy()[0]
                
                # Генерируем траекторию на несколько шагов вперед
                trajectory_points = []
                current_x, current_y = trajectory[-1]['x'], trajectory[-1]['y']
                
                for i in range(self.prediction_horizon):
                    # Простое предсказание на основе скорости
                    next_x = current_x + pred_vx * 0.1 * (i + 1)
                    next_y = current_y + pred_vy * 0.1 * (i + 1)
                    trajectory_points.append((next_x, next_y))
                
                return trajectory_points
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.linear_extrapolation(trajectory)
    
    def linear_extrapolation(self, trajectory: deque) -> List[Tuple[float, float]]:
        """Простая линейная экстраполяция"""
        if len(trajectory) < 2:
            return []
        
        # Берем последние 2 точки для вычисления направления
        p1 = trajectory[-2]
        p2 = trajectory[-1]
        
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        dt = p2['timestamp'] - p1['timestamp']
        
        if dt == 0:
            return []
        
        vx = dx / dt
        vy = dy / dt
        
        # Экстраполяция
        trajectory_points = []
        for i in range(self.prediction_horizon):
            next_x = p2['x'] + vx * 0.1 * (i + 1)
            next_y = p2['y'] + vy * 0.1 * (i + 1)
            trajectory_points.append((next_x, next_y))
        
        return trajectory_points
    
    def predict_collision_risk(self, track_id1: int, track_id2: int) -> float:
        """Предсказание риска столкновения двух пешеходов"""
        if track_id1 not in self.trajectory_buffer or track_id2 not in self.trajectory_buffer:
            return 0.0
        
        traj1 = self.predict_trajectory(track_id1)
        traj2 = self.predict_trajectory(track_id2)
        
        if not traj1 or not traj2:
            return 0.0
        
        # Проверяем минимальное расстояние в предсказанных траекториях
        min_distance = float('inf')
        for p1 in traj1:
            for p2 in traj2:
                distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                min_distance = min(min_distance, distance)
        
        # Конвертируем расстояние в риск (0-1)
        risk = max(0, 1 - min_distance / 100)  # 100 пикселей как порог
        
        return risk
    
    def get_flow_analysis(self) -> Dict:
        """Анализ потока пешеходов"""
        if not self.trajectory_buffer:
            return {}
        
        # Анализ направлений движения
        directions = []
        speeds = []
        
        for track_id, trajectory in self.trajectory_buffer.items():
            if len(trajectory) >= 2:
                p1 = trajectory[-2]
                p2 = trajectory[-1]
                
                # Направление
                dx = p2['x'] - p1['x']
                dy = p2['y'] - p1['y']
                angle = np.arctan2(dy, dx)
                directions.append(angle)
                
                # Скорость
                speed = np.sqrt(dx**2 + dy**2)
                speeds.append(speed)
        
        if not directions:
            return {}
        
        # Статистика
        mean_direction = np.mean(directions)
        mean_speed = np.mean(speeds)
        
        # Классификация направлений
        direction_categories = {
            'north': 0, 'south': 0, 'east': 0, 'west': 0,
            'northeast': 0, 'northwest': 0, 'southeast': 0, 'southwest': 0
        }
        
        for angle in directions:
            angle_deg = np.degrees(angle) % 360
            if 337.5 <= angle_deg or angle_deg < 22.5:
                direction_categories['east'] += 1
            elif 22.5 <= angle_deg < 67.5:
                direction_categories['northeast'] += 1
            elif 67.5 <= angle_deg < 112.5:
                direction_categories['north'] += 1
            elif 112.5 <= angle_deg < 157.5:
                direction_categories['northwest'] += 1
            elif 157.5 <= angle_deg < 202.5:
                direction_categories['west'] += 1
            elif 202.5 <= angle_deg < 247.5:
                direction_categories['southwest'] += 1
            elif 247.5 <= angle_deg < 292.5:
                direction_categories['south'] += 1
            else:
                direction_categories['southeast'] += 1
        
        return {
            'mean_speed': mean_speed,
            'mean_direction': mean_direction,
            'direction_distribution': direction_categories,
            'total_pedestrians': len(self.trajectory_buffer),
            'dominant_direction': max(direction_categories, key=direction_categories.get)
        }

# Утилиты для тренировки модели
def create_training_data(trajectories_file: str):
    """Создание тренировочных данных из файла траекторий"""
    # Загрузка траекторий (формат: list of trajectories)
    with open(trajectories_file, 'r') as f:
        trajectories = json.load(f)
    
    sequences = []
    targets = []
    
    for traj in trajectories:
        if len(traj) < 11:  # sequence_length + 1
            continue
        
        # Создаем последовательности
        for i in range(len(traj) - 10):
            sequence = traj[i:i+10]
            target = traj[i+10]
            
            # Конвертируем в [x, y, vx, vy]
            seq_data = []
            for j, point in enumerate(sequence):
                if j == 0:
                    vx, vy = 0, 0
                else:
                    dt = point['timestamp'] - sequence[j-1]['timestamp']
                    vx = (point['x'] - sequence[j-1]['x']) / dt if dt > 0 else 0
                    vy = (point['y'] - sequence[j-1]['y']) / dt if dt > 0 else 0
                
                seq_data.append([point['x'], point['y'], vx, vy])
            
            # Цель: следующая точка
            dt = target['timestamp'] - sequence[-1]['timestamp']
            target_vx = (target['x'] - sequence[-1]['x']) / dt if dt > 0 else 0
            target_vy = (target['y'] - sequence[-1]['y']) / dt if dt > 0 else 0
            
            sequences.append(seq_data)
            targets.append([target['x'], target['y'], target_vx, target_vy])
    
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

if __name__ == "__main__":
    # Тестирование предсказателя
    manager = TrajectoryManager()
    
    # Симуляция траектории
    track_id = 1
    for i in range(15):
        x = 100 + i * 5
        y = 200 + i * 2
        timestamp = i * 0.1
        
        manager.add_trajectory_point(track_id, x, y, timestamp)
        
        if i >= 10:
            prediction = manager.predict_trajectory(track_id)
            print(f"Step {i}: Predicted trajectory: {prediction[:3]}")
    
    # Анализ потока
    flow_analysis = manager.get_flow_analysis()
    print("Flow analysis:", flow_analysis)
