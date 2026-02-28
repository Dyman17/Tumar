import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import time
import json
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from scipy import stats
import requests

# Импорты наших модулей
from object_detection import Detection, DetectionSystem
from anomaly_detection import AnomalyResult, AnomalyDetectionSystem
from nasa_gibs import NASAGIBSClient, ThermalAnomalyAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SensorData:
    """Данные с сенсоров"""
    timestamp: float
    temperature: float
    humidity: float
    light_level: float
    pressure: Optional[float] = None
    wind_speed: Optional[float] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None

@dataclass
class EnvironmentalData:
    """Данные об окружающей среде"""
    timestamp: float
    weather_condition: str
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: float
    visibility: float
    uv_index: Optional[float] = None

@dataclass
class SatelliteData:
    """Спутниковые данные"""
    timestamp: float
    thermal_anomaly_score: float
    fire_detection_score: float
    surface_temperature: float
    ndvi_index: float
    cloud_cover: float
    data_source: str

@dataclass
class FusedData:
    """Объединенные данные"""
    timestamp: float
    local_detections: List[Detection]
    anomaly_results: List[AnomalyResult]
    sensor_data: Optional[SensorData]
    environmental_data: Optional[EnvironmentalData]
    satellite_data: Optional[SatelliteData]
    location: Optional[Tuple[float, float]]
    fusion_metadata: Dict

class WeatherAPIClient:
    """Клиент для погодного API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.cache = {}
        self.cache_timeout = 1800  # 30 минут
    
    def get_weather(self, lat: float, lon: float) -> Optional[EnvironmentalData]:
        """Получение погодных данных"""
        
        cache_key = f"{lat:.3f}_{lon:.3f}"
        current_time = time.time()
        
        # Проверяем кэш
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_timeout:
                return cached_data
        
        if not self.api_key:
            # Возвращаем симулированные данные если нет API ключа
            return self._simulate_weather(lat, lon)
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            env_data = EnvironmentalData(
                timestamp=current_time,
                weather_condition=data['weather'][0]['description'],
                temperature=data['main']['temp'],
                humidity=data['main']['humidity'],
                wind_speed=data.get('wind', {}).get('speed', 0),
                wind_direction=data.get('wind', {}).get('deg', 0),
                visibility=data.get('visibility', 10000) / 1000,  # конвертируем в км
                uv_index=None
            )
            
            # Кэшируем
            self.cache[cache_key] = (env_data, current_time)
            
            return env_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch weather data: {e}")
            return self._simulate_weather(lat, lon)
    
    def _simulate_weather(self, lat: float, lon: float) -> EnvironmentalData:
        """Симуляция погодных данных"""
        current_time = time.time()
        
        # Простая симуляция на основе времени года
        month = datetime.fromtimestamp(current_time).month
        
        if month in [12, 1, 2]:  # Зима
            temp = np.random.normal(-5, 5)
            humidity = np.random.normal(70, 10)
            condition = np.random.choice(['snow', 'cloudy', 'clear'])
        elif month in [3, 4, 5]:  # Весна
            temp = np.random.normal(15, 5)
            humidity = np.random.normal(60, 10)
            condition = np.random.choice(['rain', 'cloudy', 'clear'])
        elif month in [6, 7, 8]:  # Лето
            temp = np.random.normal(25, 5)
            humidity = np.random.normal(50, 10)
            condition = np.random.choice(['clear', 'cloudy', 'rain'])
        else:  # Осень
            temp = np.random.normal(10, 5)
            humidity = np.random.normal(65, 10)
            condition = np.random.choice(['rain', 'cloudy', 'clear'])
        
        return EnvironmentalData(
            timestamp=current_time,
            weather_condition=condition,
            temperature=temp,
            humidity=max(0, min(100, humidity)),
            wind_speed=np.random.exponential(3),
            wind_direction=np.random.uniform(0, 360),
            visibility=np.random.normal(10, 2),
            uv_index=np.random.uniform(1, 10)
        )

class DataFusionEngine:
    """Основной движок объединения данных"""
    
    def __init__(self, location: Tuple[float, float] = None, weather_api_key: str = None):
        self.location = location
        self.detection_system = DetectionSystem()
        self.anomaly_system = AnomalyDetectionSystem()
        self.nasa_client = NASAGIBSClient()
        self.thermal_analyzer = ThermalAnomalyAnalyzer(self.nasa_client)
        self.weather_client = WeatherAPIClient(weather_api_key)
        
        # История данных
        self.fusion_history = deque(maxlen=1000)
        self.sensor_history = deque(maxlen=500)
        
        # Веса для fusion
        self.fusion_weights = {
            'local_detections': 0.3,
            'anomaly_detection': 0.25,
            'satellite_thermal': 0.2,
            'satellite_fire': 0.15,
            'environmental': 0.1
        }
    
    def process_sensor_data(self, sensor_reading: Dict) -> SensorData:
        """Обработка данных с сенсоров"""
        return SensorData(
            timestamp=sensor_reading.get('timestamp', time.time()),
            temperature=sensor_reading.get('temperature', 20.0),
            humidity=sensor_reading.get('humidity', 50.0),
            light_level=sensor_reading.get('light_level', 500.0),
            pressure=sensor_reading.get('pressure'),
            wind_speed=sensor_reading.get('wind_speed'),
            gps_lat=sensor_reading.get('gps', {}).get('lat'),
            gps_lon=sensor_reading.get('gps', {}).get('lon')
        )
    
    def get_satellite_data(self, location: Tuple[float, float], 
                          timestamp: float = None) -> Optional[SatelliteData]:
        """Получение спутниковых данных"""
        
        if timestamp is None:
            timestamp = time.time()
        
        try:
            # Анализ локации
            analysis = self.thermal_analyzer.analyze_location(
                location[0], location[1], radius=0.05
            )
            
            satellite_data = SatelliteData(
                timestamp=timestamp,
                thermal_anomaly_score=analysis.get('risk_score', 0.0),
                fire_detection_score=0.0,  # TODO: добавить детекцию огня
                surface_temperature=analysis.get('detailed_analysis', {})
                                   .get('surface_temperature', {})
                                   .get('statistics', {})
                                   .get('mean_temperature', 20.0),
                ndvi_index=analysis.get('detailed_analysis', {})
                             .get('ndvi', {})
                             .get('mean_ndvi', 128.0) / 255.0,
                cloud_cover=0.3,  # TODO: получить реальные данные
                data_source="NASA_GIBS"
            )
            
            return satellite_data
            
        except Exception as e:
            logger.warning(f"Failed to get satellite data: {e}")
            return None
    
    def fuse_frame_data(self, frame: np.ndarray, 
                       sensor_data: Optional[Dict] = None) -> FusedData:
        """Объединение данных с кадра"""
        
        current_time = time.time()
        
        # Локальная детекция
        detection_results = self.detection_system.process_frame(frame)
        
        # Детекция аномалий
        anomaly_results = self.anomaly_system.process_detections(detection_results)
        
        # Обработка сенсорных данных
        processed_sensor_data = None
        if sensor_data:
            processed_sensor_data = self.process_sensor_data(sensor_data)
            self.sensor_history.append(processed_sensor_data)
        
        # Определение локации
        current_location = self.location
        if processed_sensor_data and processed_sensor_data.gps_lat:
            current_location = (processed_sensor_data.gps_lat, 
                             processed_sensor_data.gps_lon)
        
        # Погодные данные
        environmental_data = None
        if current_location:
            environmental_data = self.weather_client.get_weather(*current_location)
        
        # Спутниковые данные
        satellite_data = None
        if current_location:
            satellite_data = self.get_satellite_data(current_location, current_time)
        
        # Создаем fused data
        fused_data = FusedData(
            timestamp=current_time,
            local_detections=detection_results.get('detections', []),
            anomaly_results=anomaly_results.get('anomaly_results', []),
            sensor_data=processed_sensor_data,
            environmental_data=environmental_data,
            satellite_data=satellite_data,
            location=current_location,
            fusion_metadata={
                'frame_id': detection_results.get('frame_id'),
                'detection_summary': detection_results.get('detection_summary'),
                'track_stats': detection_results.get('track_stats'),
                'performance_stats': detection_results.get('performance_stats'),
                'anomaly_summary': anomaly_results.get('anomaly_summary')
            }
        )
        
        # Сохраняем в историю
        self.fusion_history.append(fused_data)
        
        return fused_data
    
    def calculate_comprehensive_risk(self, fused_data: FusedData) -> Dict:
        """Расчет комплексного риска"""
        
        risk_components = {}
        total_risk = 0.0
        
        # Компонент 1: Локальные детекции
        detection_count = len(fused_data.local_detections)
        detection_risk = min(detection_count / 10.0, 1.0)  # Нормализуем
        risk_components['local_detections'] = detection_risk
        total_risk += detection_risk * self.fusion_weights['local_detections']
        
        # Компонент 2: Аномалии
        anomaly_summary = fused_data.fusion_metadata.get('anomaly_summary', {})
        anomaly_risk = 1.0 if anomaly_summary.get('has_anomaly', False) else 0.0
        if anomaly_summary.get('max_severity') == 'critical':
            anomaly_risk = 1.0
        elif anomaly_summary.get('max_severity') == 'high':
            anomaly_risk = 0.8
        elif anomaly_summary.get('max_severity') == 'medium':
            anomaly_risk = 0.5
        
        risk_components['anomaly_detection'] = anomaly_risk
        total_risk += anomaly_risk * self.fusion_weights['anomaly_detection']
        
        # Компонент 3: Спутниковые тепловые данные
        if fused_data.satellite_data:
            thermal_risk = fused_data.satellite_data.thermal_anomaly_score
            fire_risk = fused_data.satellite_data.fire_detection_score
            
            risk_components['satellite_thermal'] = thermal_risk
            risk_components['satellite_fire'] = fire_risk
            
            total_risk += thermal_risk * self.fusion_weights['satellite_thermal']
            total_risk += fire_risk * self.fusion_weights['satellite_fire']
        else:
            risk_components['satellite_thermal'] = 0.0
            risk_components['satellite_fire'] = 0.0
        
        # Компонент 4: Environmental факторы
        if fused_data.environmental_data:
            env_risk = self._calculate_environmental_risk(fused_data.environmental_data)
            risk_components['environmental'] = env_risk
            total_risk += env_risk * self.fusion_weights['environmental']
        else:
            risk_components['environmental'] = 0.0
        
        # Классификация риска
        risk_level = self._classify_risk_level(total_risk)
        
        return {
            'total_risk_score': total_risk,
            'risk_level': risk_level,
            'risk_components': risk_components,
            'risk_factors': self._identify_risk_factors(risk_components),
            'confidence': self._calculate_confidence(fused_data),
            'timestamp': fused_data.timestamp
        }
    
    def _calculate_environmental_risk(self, env_data: EnvironmentalData) -> float:
        """Расчет environmental риска"""
        risk = 0.0
        
        # Экстремальные температуры
        if env_data.temperature > 35 or env_data.temperature < -10:
            risk += 0.3
        
        # Низкая видимость
        if env_data.visibility < 1.0:  # < 1 км
            risk += 0.2
        
        # Сильный ветер
        if env_data.wind_speed > 15:  # > 15 м/с
            risk += 0.2
        
        # Неблагоприятные условия
        if any(condition in env_data.weather_condition.lower() 
               for condition in ['storm', 'fog', 'snow', 'rain']):
            risk += 0.3
        
        return min(risk, 1.0)
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Классификация уровня риска"""
        if risk_score < 0.2:
            return "VERY_LOW"
        elif risk_score < 0.4:
            return "LOW"
        elif risk_score < 0.6:
            return "MEDIUM"
        elif risk_score < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _identify_risk_factors(self, risk_components: Dict) -> List[str]:
        """Идентификация факторов риска"""
        factors = []
        
        for component, score in risk_components.items():
            if score > 0.5:
                if component == 'local_detections':
                    factors.append("High object density")
                elif component == 'anomaly_detection':
                    factors.append("Anomalous activity detected")
                elif component == 'satellite_thermal':
                    factors.append("Thermal anomalies in area")
                elif component == 'satellite_fire':
                    factors.append("Fire detection from satellite")
                elif component == 'environmental':
                    factors.append("Adverse environmental conditions")
        
        return factors
    
    def _calculate_confidence(self, fused_data: FusedData) -> float:
        """Расчет уверенности в оценке"""
        confidence = 0.5  # Базовая уверенность
        
        # Наличие локальных детекций
        if fused_data.local_detections:
            confidence += 0.2
        
        # Наличие спутниковых данных
        if fused_data.satellite_data:
            confidence += 0.2
        
        # Наличие environmental данных
        if fused_data.environmental_data:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def get_fusion_statistics(self) -> Dict:
        """Получение статистики fusion"""
        if not self.fusion_history:
            return {}
        
        recent_data = list(self.fusion_history)[-100:]  # Последние 100 записей
        
        # Статистика детекций
        detection_counts = [len(d.local_detections) for d in recent_data]
        avg_detections = np.mean(detection_counts) if detection_counts else 0
        
        # Статистика аномалий
        anomaly_counts = [len(d.anomaly_results) for d in recent_data]
        avg_anomalies = np.mean(anomaly_counts) if anomaly_counts else 0
        
        # Статистика риска
        risk_scores = []
        for data in recent_data:
            risk_assessment = self.calculate_comprehensive_risk(data)
            risk_scores.append(risk_assessment['total_risk_score'])
        
        avg_risk = np.mean(risk_scores) if risk_scores else 0
        
        return {
            'total_processed': len(self.fusion_history),
            'recent_avg_detections': avg_detections,
            'recent_avg_anomalies': avg_anomalies,
            'recent_avg_risk': avg_risk,
            'data_sources': {
                'local_detections': sum(1 for d in recent_data if d.local_detections),
                'anomaly_detection': sum(1 for d in recent_data if d.anomaly_results),
                'satellite_data': sum(1 for d in recent_data if d.satellite_data),
                'environmental_data': sum(1 for d in recent_data if d.environmental_data),
                'sensor_data': sum(1 for d in recent_data if d.sensor_data)
            }
        }
    
    def export_fusion_data(self, filename: str = None) -> str:
        """Экспорт fusion данных в JSON"""
        if filename is None:
            filename = f"fusion_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Конвертируем историю в сериализуемый формат
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_records': len(self.fusion_history),
                'location': self.location
            },
            'fusion_history': []
        }
        
        for fused_data in self.fusion_history:
            record = asdict(fused_data)
            # Конвертируем детекции в словари
            record['local_detections'] = [asdict(d) for d in fused_data.local_detections]
            record['anomaly_results'] = [asdict(r) for r in fused_data.anomaly_results]
            
            if fused_data.sensor_data:
                record['sensor_data'] = asdict(fused_data.sensor_data)
            if fused_data.environmental_data:
                record['environmental_data'] = asdict(fused_data.environmental_data)
            if fused_data.satellite_data:
                record['satellite_data'] = asdict(fused_data.satellite_data)
            
            export_data['fusion_history'].append(record)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filename

if __name__ == "__main__":
    # Тестирование системы fusion
    print("Testing Data Fusion Engine...")
    
    # Инициализация (Москва)
    location = (55.7558, 37.6173)
    fusion_engine = DataFusionEngine(location=location)
    
    # Симуляция обработки кадра
    try:
        # Создаем тестовый кадр
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Тестовые сенсорные данные
        sensor_data = {
            'timestamp': time.time(),
            'temperature': 25.5,
            'humidity': 60.0,
            'light_level': 800,
            'gps': {'lat': location[0], 'lon': location[1]}
        }
        
        # Обработка
        fused_result = fusion_engine.fuse_frame_data(test_frame, sensor_data)
        
        # Расчет риска
        risk_assessment = fusion_engine.calculate_comprehensive_risk(fused_result)
        
        print(f"Fusion completed successfully!")
        print(f"Risk Level: {risk_assessment['risk_level']}")
        print(f"Risk Score: {risk_assessment['total_risk_score']:.3f}")
        print(f"Risk Factors: {risk_assessment['risk_factors']}")
        print(f"Confidence: {risk_assessment['confidence']:.3f}")
        
        # Статистика
        stats = fusion_engine.get_fusion_statistics()
        print(f"\nFusion Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Экспорт данных
        export_file = fusion_engine.export_fusion_data()
        print(f"\nData exported to: {export_file}")
        
    except Exception as e:
        print(f"Error during fusion testing: {e}")
        logger.exception("Fusion test failed")
