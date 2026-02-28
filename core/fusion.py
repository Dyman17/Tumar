"""
Fusion Engine - Объединение всех данных
FusedVector = [convoy_flag, anomaly_speed, anomaly_direction, density, thermal_activity]
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

from tracker import Track
from behavior import BehaviorFeatures
from thermal import ThermalData

@dataclass
class FusedFeatures:
    """Класс для объединенных признаков"""
    convoy_flag: bool
    convoy_size: int
    anomaly_speed: bool
    anomaly_direction: bool
    density: float
    thermal_activity: bool
    thermal_data_available: bool
    timestamp: float

class FusionEngine:
    """Движок объединения данных"""
    
    def __init__(self):
        # Веса для различных признаков (будут использоваться в Risk Engine)
        self.feature_weights = {
            'convoy_flag': 0.35,
            'anomaly_speed': 0.20,
            'anomaly_direction': 0.15,
            'density': 0.15,
            'thermal_activity': 0.15
        }
        
        # История объединенных данных
        self.fusion_history = []
        self.max_history_length = 1000
        
        # Пороги
        self.density_threshold = 0.01  # объектов на пиксель
        
    def fuse_data(self, tracks: List[Track], behavior_features: BehaviorFeatures, 
                  thermal_data: ThermalData, location: Optional[Tuple[float, float]] = None) -> FusedFeatures:
        """Объединение всех данных в единый вектор признаков"""
        
        current_time = time.time()
        
        # 1. Группировка техники (из behavior)
        convoy_flag = behavior_features.convoy_flag
        convoy_size = behavior_features.convoy_size
        
        # 2. Аномальная скорость (из behavior)
        anomaly_speed = behavior_features.anomaly_speed
        
        # 3. Резкое изменение направления (из behavior)
        anomaly_direction = behavior_features.anomaly_direction
        
        # 4. Плотность (из behavior)
        density = behavior_features.density
        
        # 5. Тепловая активность (из thermal)
        # Проверяем есть ли техника в зонах с тепловой активностью
        vehicle_locations = [(int(track.center[0]), int(track.center[1])) 
                           for track in tracks 
                           if track.class_name in ['car', 'truck', 'bus']]
        
        thermal_activity = False
        if thermal_data.data_available:
            # Импортируем здесь для избежания циклических зависимостей
            from thermal import ThermalIntegration
            thermal_integration = ThermalIntegration()
            thermal_activity = thermal_integration.check_thermal_activity_in_zones(
                thermal_data, vehicle_locations
            )
        
        # Создаем объединенный вектор признаков
        fused_features = FusedFeatures(
            convoy_flag=convoy_flag,
            convoy_size=convoy_size,
            anomaly_speed=anomaly_speed,
            anomaly_direction=anomaly_direction,
            density=density,
            thermal_activity=thermal_activity,
            thermal_data_available=thermal_data.data_available,
            timestamp=current_time
        )
        
        # Сохраняем в историю
        self.fusion_history.append(fused_features)
        if len(self.fusion_history) > self.max_history_length:
            self.fusion_history.pop(0)
        
        return fused_features
    
    def get_feature_vector(self, fused_features: FusedFeatures) -> List[float]:
        """Получение числового вектора признаков для Risk Engine"""
        
        return [
            float(fused_features.convoy_flag),
            float(fused_features.anomaly_speed),
            float(fused_features.anomaly_direction),
            float(fused_features.density),
            float(fused_features.thermal_activity)
        ]
    
    def get_feature_summary(self, fused_features: FusedFeatures) -> Dict:
        """Получение сводки признаков"""
        
        feature_vector = self.get_feature_vector(fused_features)
        
        # Вычисляем взвешенную сумму (предварительный риск)
        weighted_sum = sum(feature * weight 
                          for feature, weight in zip(feature_vector, 
                                                    self.feature_weights.values()))
        
        return {
            'feature_vector': feature_vector,
            'weighted_sum': weighted_sum,
            'convoy_detected': fused_features.convoy_flag,
            'convoy_size': fused_features.convoy_size,
            'speed_anomaly': fused_features.anomaly_speed,
            'direction_anomaly': fused_features.anomaly_direction,
            'density_level': fused_features.density,
            'thermal_activity': fused_features.thermal_activity,
            'thermal_data_available': fused_features.thermal_data_available,
            'timestamp': fused_features.timestamp
        }
    
    def analyze_trends(self, time_window: float = 300.0) -> Dict:
        """Анализ трендов в объединенных данных"""
        
        current_time = time.time()
        recent_data = [f for f in self.fusion_history 
                      if current_time - f.timestamp <= time_window]
        
        if len(recent_data) < 2:
            return {'trend': 'insufficient_data'}
        
        # Анализируем тренды каждого признака
        trends = {}
        
        for feature_name in ['convoy_flag', 'anomaly_speed', 'anomaly_direction', 
                           'thermal_activity']:
            values = [float(getattr(f, feature_name)) for f in recent_data]
            
            # Простой тренд - сравниваем среднее значение первой и второй половины
            mid_point = len(values) // 2
            first_half_avg = sum(values[:mid_point]) / mid_point if mid_point > 0 else 0
            second_half_avg = sum(values[mid_point:]) / (len(values) - mid_point) if len(values) - mid_point > 0 else 0
            
            if second_half_avg > first_half_avg + 0.1:
                trends[feature_name] = 'increasing'
            elif second_half_avg < first_half_avg - 0.1:
                trends[feature_name] = 'decreasing'
            else:
                trends[feature_name] = 'stable'
        
        # Анализ плотности
        density_values = [f.density for f in recent_data]
        mid_point = len(density_values) // 2
        first_half_density = sum(density_values[:mid_point]) / mid_point if mid_point > 0 else 0
        second_half_density = sum(density_values[mid_point:]) / (len(density_values) - mid_point) if len(density_values) - mid_point > 0 else 0
        
        if second_half_density > first_half_density * 1.2:
            trends['density'] = 'increasing'
        elif second_half_density < first_half_density * 0.8:
            trends['density'] = 'decreasing'
        else:
            trends['density'] = 'stable'
        
        return {
            'trend': 'analyzed',
            'feature_trends': trends,
            'data_points': len(recent_data),
            'time_window': time_window
        }
    
    def get_statistics(self) -> Dict:
        """Получение статистики объединенных данных"""
        
        if not self.fusion_history:
            return {}
        
        # Общая статистика
        total_records = len(self.fusion_history)
        
        # Статистика по признакам
        convoy_count = sum(1 for f in self.fusion_history if f.convoy_flag)
        speed_anomaly_count = sum(1 for f in self.fusion_history if f.anomaly_speed)
        direction_anomaly_count = sum(1 for f in self.fusion_history if f.anomaly_direction)
        thermal_activity_count = sum(1 for f in self.fusion_history if f.thermal_activity)
        
        # Средняя плотность
        avg_density = sum(f.density for f in self.fusion_history) / total_records
        
        # Распределение размеров конвоев
        convoy_sizes = [f.convoy_size for f in self.fusion_history if f.convoy_flag]
        avg_convoy_size = sum(convoy_sizes) / len(convoy_sizes) if convoy_sizes else 0
        
        return {
            'total_records': total_records,
            'convoy_detections': convoy_count,
            'speed_anomalies': speed_anomaly_count,
            'direction_anomalies': direction_anomaly_count,
            'thermal_activities': thermal_activity_count,
            'convoy_rate': convoy_count / total_records,
            'speed_anomaly_rate': speed_anomaly_count / total_records,
            'direction_anomaly_rate': direction_anomaly_count / total_records,
            'thermal_activity_rate': thermal_activity_count / total_records,
            'average_density': avg_density,
            'average_convoy_size': avg_convoy_size,
            'thermal_data_availability': sum(1 for f in self.fusion_history if f.thermal_data_available) / total_records
        }
    
    def export_fusion_data(self, filename: str = None) -> str:
        """Экспорт объединенных данных"""
        
        if filename is None:
            filename = f"fusion_data_{int(time.time())}.json"
        
        import json
        
        export_data = {
            'metadata': {
                'export_timestamp': time.time(),
                'total_records': len(self.fusion_history),
                'feature_weights': self.feature_weights
            },
            'fusion_history': [
                {
                    'convoy_flag': f.convoy_flag,
                    'convoy_size': f.convoy_size,
                    'anomaly_speed': f.anomaly_speed,
                    'anomaly_direction': f.anomaly_direction,
                    'density': f.density,
                    'thermal_activity': f.thermal_activity,
                    'thermal_data_available': f.thermal_data_available,
                    'timestamp': f.timestamp
                }
                for f in self.fusion_history
            ],
            'statistics': self.get_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename

if __name__ == "__main__":
    # Тестирование Fusion Engine
    from tracker import Track
    from behavior import BehaviorFeatures
    from thermal import ThermalData
    
    fusion_engine = FusionEngine()
    
    print("Testing Fusion Engine...")
    
    # Создаем тестовые данные
    test_tracks = [
        Track(
            id=1, class_name='truck', center=(100, 100), velocity=(10, 5),
            acceleration=(0, 0), bbox=(80, 80, 120, 120), confidence=0.8,
            last_seen=time.time(), trajectory=[], disappeared_count=0,
            kalman_filter=None
        )
    ]
    
    test_behavior = BehaviorFeatures(
        convoy_flag=True, convoy_size=3, anomaly_speed=False,
        anomaly_direction=False, density=0.02, grid_density={},
        avg_scene_speed=15.0, speed_variance=5.0
    )
    
    test_thermal = ThermalData(
        timestamp=time.time(), thermal_anomaly_score=0.6,
        fire_detection_score=0.1, surface_temperature=35.0,
        hotspot_locations=[(50, 50)], data_available=True
    )
    
    try:
        # Тест объединения данных
        fused_features = fusion_engine.fuse_data(
            test_tracks, test_behavior, test_thermal
        )
        
        print(f"Fused Features:")
        print(f"  Convoy Flag: {fused_features.convoy_flag}")
        print(f"  Convoy Size: {fused_features.convoy_size}")
        print(f"  Anomaly Speed: {fused_features.anomaly_speed}")
        print(f"  Anomaly Direction: {fused_features.anomaly_direction}")
        print(f"  Density: {fused_features.density:.3f}")
        print(f"  Thermal Activity: {fused_features.thermal_activity}")
        print(f"  Thermal Data Available: {fused_features.thermal_data_available}")
        
        # Получение вектора признаков
        feature_vector = fusion_engine.get_feature_vector(fused_features)
        print(f"\nFeature Vector: {feature_vector}")
        
        # Сводка признаков
        summary = fusion_engine.get_feature_summary(fused_features)
        print(f"\nFeature Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Статистика
        stats = fusion_engine.get_statistics()
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Экспорт данных
        export_file = fusion_engine.export_fusion_data()
        print(f"\nData exported to: {export_file}")
        
        print("\nFusion Engine test completed successfully!")
        
    except Exception as e:
        print(f"Error during fusion engine test: {e}")
        import traceback
        traceback.print_exc()
