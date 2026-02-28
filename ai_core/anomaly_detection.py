import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import defaultdict, deque
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AnomalyResult:
    """Результат детекции аномалии"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    confidence: float
    description: str
    timestamp: float
    severity: str  # 'low', 'medium', 'high', 'critical'

class BaselineModel:
    """Модель базовой активности"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.activity_history = deque(maxlen=window_size)
        self.baseline_stats = {}
        self.is_trained = False
        
    def add_observation(self, observation: Dict):
        """Добавление наблюдения в историю"""
        self.activity_history.append({
            'timestamp': observation.get('timestamp', time.time()),
            'object_count': observation.get('object_count', 0),
            'class_distribution': observation.get('class_distribution', {}),
            'movement_patterns': observation.get('movement_patterns', []),
            'spatial_density': observation.get('spatial_density', 0.0),
            'environmental': observation.get('environmental', {})
        })
        
        # Обновляем baseline если накоплено достаточно данных
        if len(self.activity_history) >= self.window_size and not self.is_trained:
            self.update_baseline()
    
    def update_baseline(self):
        """Обновление baseline статистики"""
        if len(self.activity_history) < 10:
            return
        
        # Извлекаем признаки
        object_counts = [obs['object_count'] for obs in self.activity_history]
        spatial_densities = [obs['spatial_density'] for obs in self.activity_history]
        
        # Считаем статистики
        self.baseline_stats = {
            'object_count': {
                'mean': np.mean(object_counts),
                'std': np.std(object_counts),
                'min': np.min(object_counts),
                'max': np.max(object_counts),
                'percentile_95': np.percentile(object_counts, 95)
            },
            'spatial_density': {
                'mean': np.mean(spatial_densities),
                'std': np.std(spatial_densities),
                'min': np.min(spatial_densities),
                'max': np.max(spatial_densities)
            }
        }
        
        # Анализ распределения классов
        class_distributions = [obs['class_distribution'] for obs in self.activity_history]
        self.baseline_stats['class_patterns'] = self.analyze_class_patterns(class_distributions)
        
        self.is_trained = True
    
    def analyze_class_patterns(self, class_distributions: List[Dict]) -> Dict:
        """Анализ паттернов классов"""
        if not class_distributions:
            return {}
        
        # Собираем все классы
        all_classes = set()
        for dist in class_distributions:
            all_classes.update(dist.keys())
        
        patterns = {}
        for class_name in all_classes:
            counts = [dist.get(class_name, 0) for dist in class_distributions]
            patterns[class_name] = {
                'mean': np.mean(counts),
                'std': np.std(counts),
                'frequency': sum(1 for c in counts if c > 0) / len(counts)
            }
        
        return patterns
    
    def get_baseline_stats(self) -> Dict:
        """Получение baseline статистики"""
        return self.baseline_stats

class AnomalyDetector:
    """Основной детектор аномалий"""
    
    def __init__(self, contamination: float = 0.1):
        self.baseline_model = BaselineModel()
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Сохраняем 95% дисперсии
        
        self.anomaly_history = deque(maxlen=1000)
        self.is_trained = False
        
        # Пороги для разных типов аномалий
        self.thresholds = {
            'statistical': 0.05,  # 5% уровень значимости
            'isolation': -0.1,    # Порог для Isolation Forest
            'density_change': 2.0, # 2 стандартных отклонения
            'temporal_pattern': 3.0 # 3 стандартных отклонения
        }
    
    def extract_features(self, observation: Dict) -> np.ndarray:
        """Извлечение признаков из наблюдения"""
        features = []
        
        # Количественные признаки
        features.append(observation.get('object_count', 0))
        features.append(observation.get('spatial_density', 0.0))
        
        # Признаки распределения классов
        class_dist = observation.get('class_distribution', {})
        common_classes = ['person', 'car', 'truck', 'animal', 'fire']
        for class_name in common_classes:
            features.append(class_dist.get(class_name, 0))
        
        # Временные признаки
        timestamp = observation.get('timestamp', time.time())
        hour = time.localtime(timestamp).tm_hour
        features.append(hour)
        
        # Признаки движения (если есть)
        movement_patterns = observation.get('movement_patterns', [])
        if movement_patterns:
            avg_speed = np.mean([p.get('speed', 0) for p in movement_patterns])
            features.append(avg_speed)
        else:
            features.append(0)
        
        # Environmental признаки
        env = observation.get('environmental', {})
        features.append(env.get('temperature', 20))
        features.append(env.get('humidity', 50))
        features.append(env.get('light_level', 500))
        
        return np.array(features)
    
    def train_model(self, observations: List[Dict]):
        """Тренировка модели на исторических данных"""
        if len(observations) < 50:
            return False
        
        # Извлекаем признаки
        features = []
        for obs in observations:
            feature_vec = self.extract_features(obs)
            features.append(feature_vec)
        
        features = np.array(features)
        
        # Масштабирование
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)
        
        # PCA для уменьшения размерности
        self.pca.fit(scaled_features)
        pca_features = self.pca.transform(scaled_features)
        
        # Тренировка Isolation Forest
        self.isolation_forest.fit(pca_features)
        
        self.is_trained = True
        return True
    
    def detect_statistical_anomaly(self, observation: Dict) -> AnomalyResult:
        """Статистическая детекция аномалий"""
        if not self.baseline_model.is_trained:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type="statistical",
                confidence=0.0,
                description="Baseline not trained",
                timestamp=time.time(),
                severity="low"
            )
        
        baseline = self.baseline_model.get_baseline_stats()
        anomalies = []
        
        # Проверка количества объектов
        obj_count = observation.get('object_count', 0)
        if 'object_count' in baseline:
            mean = baseline['object_count']['mean']
            std = baseline['object_count']['std']
            z_score = abs((obj_count - mean) / (std + 1e-8))
            
            if z_score > self.thresholds['density_change']:
                anomalies.append(f"Unusual object count: {obj_count} (expected: {mean:.1f}±{std:.1f})")
        
        # Проверка плотности
        density = observation.get('spatial_density', 0.0)
        if 'spatial_density' in baseline:
            mean = baseline['spatial_density']['mean']
            std = baseline['spatial_density']['std']
            z_score = abs((density - mean) / (std + 1e-8))
            
            if z_score > self.thresholds['density_change']:
                anomalies.append(f"Unusual spatial density: {density:.3f}")
        
        # Проверка паттернов классов
        class_dist = observation.get('class_distribution', {})
        if 'class_patterns' in baseline:
            for class_name, pattern in baseline['class_patterns'].items():
                count = class_dist.get(class_name, 0)
                if pattern['frequency'] > 0.1 and count > pattern['mean'] + 2 * pattern['std']:
                    anomalies.append(f"Unusual {class_name} count: {count}")
        
        is_anomaly = len(anomalies) > 0
        anomaly_score = min(len(anomalies) * 0.2, 1.0)
        severity = "low" if anomaly_score < 0.4 else "medium" if anomaly_score < 0.7 else "high"
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_type="statistical",
            confidence=0.8 if is_anomaly else 0.9,
            description="; ".join(anomalies) if anomalies else "Normal activity",
            timestamp=time.time(),
            severity=severity
        )
    
    def detect_isolation_anomaly(self, observation: Dict) -> AnomalyResult:
        """Детекция аномалий с помощью Isolation Forest"""
        if not self.is_trained:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type="isolation",
                confidence=0.0,
                description="Model not trained",
                timestamp=time.time(),
                severity="low"
            )
        
        features = self.extract_features(observation).reshape(1, -1)
        
        # Масштабирование и PCA
        scaled_features = self.scaler.transform(features)
        pca_features = self.pca.transform(scaled_features)
        
        # Предсказание
        anomaly_score = self.isolation_forest.decision_function(pca_features)[0]
        is_anomaly = anomaly_score < self.thresholds['isolation']
        
        severity = "low" if anomaly_score > -0.3 else "medium" if anomaly_score > -0.6 else "high"
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=float(abs(anomaly_score)),
            anomaly_type="isolation",
            confidence=0.7,
            description=f"Isolation score: {anomaly_score:.3f}",
            timestamp=time.time(),
            severity=severity
        )
    
    def detect_temporal_anomaly(self, observations: List[Dict]) -> AnomalyResult:
        """Детекция временных аномалий"""
        if len(observations) < 10:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type="temporal",
                confidence=0.0,
                description="Insufficient temporal data",
                timestamp=time.time(),
                severity="low"
            )
        
        # Анализ временных рядов
        recent_observations = observations[-10:]
        object_counts = [obs.get('object_count', 0) for obs in recent_observations]
        
        # Проверка на резкие изменения
        if len(object_counts) >= 2:
            recent_change = abs(object_counts[-1] - object_counts[-2])
            historical_std = np.std(object_counts[:-1]) if len(object_counts) > 2 else 1
            
            if recent_change > self.thresholds['temporal_pattern'] * historical_std:
                return AnomalyResult(
                    is_anomaly=True,
                    anomaly_score=min(recent_change / (historical_std + 1e-8), 1.0),
                    anomaly_type="temporal",
                    confidence=0.8,
                    description=f"Sudden change in object count: {recent_change}",
                    timestamp=time.time(),
                    severity="medium"
                )
        
        return AnomalyResult(
            is_anomaly=False,
            anomaly_score=0.0,
            anomaly_type="temporal",
            confidence=0.9,
            description="Normal temporal pattern",
            timestamp=time.time(),
            severity="low"
        )
    
    def detect_anomaly(self, observation: Dict, historical_observations: List[Dict] = None) -> List[AnomalyResult]:
        """Комплексная детекция аномалий"""
        results = []
        
        # Обновляем baseline
        self.baseline_model.add_observation(observation)
        
        # Тренируем модель если нужно
        if historical_observations and len(historical_observations) >= 50 and not self.is_trained:
            self.train_model(historical_observations)
        
        # Статистическая детекция
        stat_result = self.detect_statistical_anomaly(observation)
        results.append(stat_result)
        
        # Isolation Forest
        iso_result = self.detect_isolation_anomaly(observation)
        results.append(iso_result)
        
        # Временная детекция
        if historical_observations:
            temp_result = self.detect_temporal_anomaly(historical_observations[-20:])
            results.append(temp_result)
        
        # Сохраняем результаты
        self.anomaly_history.append({
            'timestamp': time.time(),
            'observation': observation,
            'results': results
        })
        
        return results
    
    def get_anomaly_summary(self, results: List[AnomalyResult]) -> Dict:
        """Получение сводки аномалий"""
        anomaly_count = sum(1 for r in results if r.is_anomaly)
        max_severity = "low"
        
        for result in results:
            if result.is_anomaly:
                if result.severity == "critical":
                    max_severity = "critical"
                    break
                elif result.severity == "high" and max_severity != "critical":
                    max_severity = "high"
                elif result.severity == "medium" and max_severity in ["low", "medium"]:
                    max_severity = "medium"
        
        return {
            'has_anomaly': anomaly_count > 0,
            'anomaly_count': anomaly_count,
            'max_severity': max_severity,
            'total_checks': len(results),
            'anomaly_types': [r.anomaly_type for r in results if r.is_anomaly]
        }
    
    def get_anomaly_statistics(self) -> Dict:
        """Статистика аномалий"""
        if not self.anomaly_history:
            return {}
        
        total_checks = len(self.anomaly_history)
        anomaly_periods = sum(1 for h in self.anomaly_history 
                            if any(r.is_anomaly for r in h['results']))
        
        return {
            'total_checks': total_checks,
            'anomaly_periods': anomaly_periods,
            'anomaly_rate': anomaly_periods / total_checks if total_checks > 0 else 0,
            'model_trained': self.is_trained,
            'baseline_trained': self.baseline_model.is_trained
        }

# Интеграция с системой детекции
class AnomalyDetectionSystem:
    """Система аномалий для интеграции с детекцией"""
    
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.observation_history = deque(maxlen=1000)
        
    def process_detections(self, detection_results: Dict) -> Dict:
        """Обработка результатов детекции на аномалии"""
        # Подготовка наблюдения
        observation = {
            'timestamp': detection_results.get('timestamp', time.time()),
            'object_count': sum(detection_results.get('detection_summary', {}).values()),
            'class_distribution': detection_results.get('detection_summary', {}),
            'spatial_density': self.calculate_spatial_density(detection_results.get('detections', [])),
            'movement_patterns': [],  # TODO: добавить анализ движения
            'environmental': detection_results.get('environmental', {})
        }
        
        # Добавляем в историю
        self.observation_history.append(observation)
        
        # Детекция аномалий
        anomaly_results = self.anomaly_detector.detect_anomaly(
            observation, 
            list(self.observation_history)
        )
        
        # Сводка
        anomaly_summary = self.anomaly_detector.get_anomaly_summary(anomaly_results)
        
        return {
            'observation': observation,
            'anomaly_results': anomaly_results,
            'anomaly_summary': anomaly_summary,
            'statistics': self.anomaly_detector.get_anomaly_statistics()
        }
    
    def calculate_spatial_density(self, detections: List) -> float:
        """Расчет пространственной плотности"""
        if not detections:
            return 0.0
        
        # Простая метрика: количество объектов на единицу площади
        # В реальной системе нужно использовать реальные координаты
        return len(detections) / 1000.0  # Нормализованная плотность

if __name__ == "__main__":
    # Тестирование системы
    anomaly_system = AnomalyDetectionSystem()
    
    # Симуляция данных
    for i in range(100):
        # Нормальные данные
        if i < 80:
            observation = {
                'timestamp': time.time(),
                'object_count': np.random.poisson(5),
                'class_distribution': {
                    'person': np.random.poisson(3),
                    'car': np.random.poisson(2)
                },
                'spatial_density': np.random.normal(0.01, 0.005),
                'environmental': {
                    'temperature': 20 + np.random.normal(0, 2),
                    'humidity': 50 + np.random.normal(0, 10)
                }
            }
        else:
            # Аномальные данные
            observation = {
                'timestamp': time.time(),
                'object_count': np.random.poisson(20),  # Резкое увеличение
                'class_distribution': {
                    'person': np.random.poisson(15),
                    'car': np.random.poisson(5)
                },
                'spatial_density': np.random.normal(0.05, 0.01),
                'environmental': {
                    'temperature': 20 + np.random.normal(0, 2),
                    'humidity': 50 + np.random.normal(0, 10)
                }
            }
        
        results = anomaly_system.anomaly_detector.detect_anomaly(
            observation, 
            list(anomaly_system.observation_history)
        )
        
        summary = anomaly_system.anomaly_detector.get_anomaly_summary(results)
        
        if summary['has_anomaly']:
            print(f"Anomaly detected at step {i}: {summary['max_severity']} severity")
            for result in results:
                if result.is_anomaly:
                    print(f"  - {result.anomaly_type}: {result.description}")
        
        anomaly_system.observation_history.append(observation)
    
    print("\nFinal statistics:")
    stats = anomaly_system.anomaly_detector.get_anomaly_statistics()
    print(json.dumps(stats, indent=2))
