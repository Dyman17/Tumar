"""
Thermal Integration Layer - Интеграция с тепловыми данными NASA GIBS
Не realtime 30 fps, а слой аномалий
"""

import numpy as np
import requests
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import time
import logging
from urllib.parse import urlencode
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThermalData:
    """Класс для хранения тепловых данных"""
    timestamp: float
    thermal_anomaly_score: float
    fire_detection_score: float
    surface_temperature: float
    hotspot_locations: List[Tuple[int, int]]
    data_available: bool

class ThermalIntegration:
    """Интеграция с тепловыми данными"""
    
    def __init__(self):
        # NASA GIBS API
        self.base_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
        
        # Слои для анализа
        self.thermal_layers = [
            'MODIS_Terra_Thermal_Anomalies_Day',
            'MODIS_Aqua_Thermal_Anomalies_Day'
        ]
        
        self.fire_layers = [
            'MODIS_Terra_Fire_Detection_Day',
            'MODIS_Aqua_Fire_Detection_Day'
        ]
        
        # Пороги
        self.thermal_threshold = 0.5
        self.fire_threshold = 0.3
        self.temperature_threshold = 40.0  # градусов
        
        # Кэш для данных
        self.cache = {}
        self.cache_timeout = 3600  # 1 час
        
        # История данных
        self.thermal_history = []
        self.max_history_length = 100
        
    def get_thermal_data(self, location: Tuple[float, float], 
                        radius: float = 0.05) -> ThermalData:
        """Получение тепловых данных для локации"""
        
        current_time = time.time()
        cache_key = f"{location[0]:.3f}_{location[1]:.3f}_{radius:.3f}"
        
        # Проверяем кэш
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_timeout:
                return cached_data
        
        try:
            # Получаем данные от NASA
            thermal_data = self._fetch_nasa_thermal_data(location, radius)
            
            # Сохраняем в кэш
            self.cache[cache_key] = (thermal_data, current_time)
            
            # Добавляем в историю
            self.thermal_history.append(thermal_data)
            if len(self.thermal_history) > self.max_history_length:
                self.thermal_history.pop(0)
            
            return thermal_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch thermal data: {e}")
            # Возвращаем пустые данные при ошибке
            return ThermalData(
                timestamp=current_time,
                thermal_anomaly_score=0.0,
                fire_detection_score=0.0,
                surface_temperature=20.0,
                hotspot_locations=[],
                data_available=False
            )
    
    def _fetch_nasa_thermal_data(self, location: Tuple[float, float], 
                                radius: float) -> ThermalData:
        """Получение данных от NASA GIBS"""
        
        # Формируем bbox
        lat, lon = location
        bbox = (lon - radius, lat - radius, lon + radius, lat + radius)
        
        # Текущая дата
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Получаем тепловые аномалии
        thermal_score = self._get_thermal_anomaly_score(bbox, date_str)
        
        # Получаем детекцию огня
        fire_score = self._get_fire_detection_score(bbox, date_str)
        
        # Получаем температуру поверхности
        surface_temp = self._get_surface_temperature(bbox, date_str)
        
        # Определяем горячие точки
        hotspots = self._detect_hotspots(bbox, date_str)
        
        return ThermalData(
            timestamp=time.time(),
            thermal_anomaly_score=thermal_score,
            fire_detection_score=fire_score,
            surface_temperature=surface_temp,
            hotspot_locations=hotspots,
            data_available=True
        )
    
    def _get_thermal_anomaly_score(self, bbox: Tuple[float, float, float, float], 
                                  date_str: str) -> float:
        """Получение оценки тепловых аномалий"""
        
        for layer in self.thermal_layers:
            try:
                # Запрос к NASA GIBS
                params = {
                    'SERVICE': 'WMS',
                    'VERSION': '1.3.0',
                    'REQUEST': 'GetMap',
                    'LAYERS': layer,
                    'FORMAT': 'image/png',
                    'TRANSPARENT': 'TRUE',
                    'BBOX': f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",
                    'WIDTH': 64,
                    'HEIGHT': 64,
                    'TIME': date_str
                }
                
                url = f"{self.base_url}?{urlencode(params)}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    # Анализ изображения для определения аномалий
                    image_data = np.frombuffer(response.content, np.uint8)
                    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        # Конвертируем в градации серого
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        
                        # Анализируем пиксели
                        anomaly_pixels = np.sum(gray > 128)  # яркие пиксели
                        total_pixels = gray.size
                        
                        anomaly_score = anomaly_pixels / total_pixels
                        return min(anomaly_score * 5, 1.0)  # нормализуем
                
            except Exception as e:
                logger.warning(f"Failed to fetch thermal layer {layer}: {e}")
                continue
        
        return 0.0
    
    def _get_fire_detection_score(self, bbox: Tuple[float, float, float, float], 
                                 date_str: str) -> float:
        """Получение оценки детекции огня"""
        
        for layer in self.fire_layers:
            try:
                params = {
                    'SERVICE': 'WMS',
                    'VERSION': '1.3.0',
                    'REQUEST': 'GetMap',
                    'LAYERS': layer,
                    'FORMAT': 'image/png',
                    'TRANSPARENT': 'TRUE',
                    'BBOX': f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",
                    'WIDTH': 64,
                    'HEIGHT': 64,
                    'TIME': date_str
                }
                
                url = f"{self.base_url}?{urlencode(params)}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    image_data = np.frombuffer(response.content, np.uint8)
                    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        # Анализ красного канала (огонь обычно красный)
                        red_channel = image[:, :, 2]
                        fire_pixels = np.sum(red_channel > 200)
                        total_pixels = red_channel.size
                        
                        fire_score = fire_pixels / total_pixels
                        return min(fire_score * 10, 1.0)  # нормализуем
                
            except Exception as e:
                logger.warning(f"Failed to fetch fire layer {layer}: {e}")
                continue
        
        return 0.0
    
    def _get_surface_temperature(self, bbox: Tuple[float, float, float, float], 
                                date_str: str) -> float:
        """Получение температуры поверхности"""
        
        try:
            # Используем слой температуры поверхности
            layer = 'MODIS_Terra_LST_Day'
            
            params = {
                'SERVICE': 'WMS',
                'VERSION': '1.3.0',
                'REQUEST': 'GetMap',
                'LAYERS': layer,
                'FORMAT': 'image/png',
                'BBOX': f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",
                'WIDTH': 32,
                'HEIGHT': 32,
                'TIME': date_str
            }
            
            url = f"{self.base_url}?{urlencode(params)}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                image_data = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                
                if image is not None:
                    # Простая оценка температуры на основе яркости
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    avg_brightness = np.mean(gray)
                    
                    # Конвертируем яркость в температуру (упрощенно)
                    # 0-255 -> -20°C до 50°C
                    temperature = -20 + (avg_brightness / 255) * 70
                    
                    return temperature
        
        except Exception as e:
            logger.warning(f"Failed to fetch surface temperature: {e}")
        
        return 20.0  # температура по умолчанию
    
    def _detect_hotspots(self, bbox: Tuple[float, float, float, float], 
                        date_str: str) -> List[Tuple[int, int]]:
        """Детекция горячих точек"""
        
        hotspots = []
        
        try:
            # Используем слой тепловых аномалий
            layer = 'MODIS_Terra_Thermal_Anomalies_Day'
            
            params = {
                'SERVICE': 'WMS',
                'VERSION': '1.3.0',
                'REQUEST': 'GetMap',
                'LAYERS': layer,
                'FORMAT': 'image/png',
                'BBOX': f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",
                'WIDTH': 128,
                'HEIGHT': 128,
                'TIME': date_str
            }
            
            url = f"{self.base_url}?{urlencode(params)}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                image_data = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                
                if image is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    
                    # Находим контуры ярких областей
                    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                                  cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if cv2.contourArea(contour) > 10:  # минимальный размер
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                hotspots.append((cx, cy))
        
        except Exception as e:
            logger.warning(f"Failed to detect hotspots: {e}")
        
        return hotspots
    
    def check_thermal_activity_in_zones(self, thermal_data: ThermalData, 
                                       vehicle_locations: List[Tuple[int, int]]) -> bool:
        """Проверка тепловой активности в зонах с техникой"""
        
        if not thermal_data.data_available or not vehicle_locations:
            return False
        
        # Проверяем пороги
        if (thermal_data.thermal_anomaly_score > self.thermal_threshold or
            thermal_data.fire_detection_score > self.fire_threshold or
            thermal_data.surface_temperature > self.temperature_threshold):
            
            # Дополнительная проверка - есть ли горячие точки рядом с техникой
            for vehicle_loc in vehicle_locations:
                for hotspot in thermal_data.hotspot_locations:
                    distance = np.linalg.norm(
                        np.array(vehicle_loc) - np.array(hotspot)
                    )
                    
                    if distance < 50:  # 50 пикселей
                        return True
        
        return False
    
    def get_thermal_summary(self, thermal_data: ThermalData) -> Dict:
        """Получение сводки тепловых данных"""
        
        return {
            'data_available': thermal_data.data_available,
            'thermal_anomaly_score': thermal_data.thermal_anomaly_score,
            'fire_detection_score': thermal_data.fire_detection_score,
            'surface_temperature': thermal_data.surface_temperature,
            'hotspot_count': len(thermal_data.hotspot_locations),
            'risk_level': self._classify_thermal_risk(thermal_data)
        }
    
    def _classify_thermal_risk(self, thermal_data: ThermalData) -> str:
        """Классификация теплового риска"""
        
        if not thermal_data.data_available:
            return "unknown"
        
        # Комплексная оценка риска
        risk_score = 0.0
        
        if thermal_data.thermal_anomaly_score > self.thermal_threshold:
            risk_score += 0.4
        
        if thermal_data.fire_detection_score > self.fire_threshold:
            risk_score += 0.4
        
        if thermal_data.surface_temperature > self.temperature_threshold:
            risk_score += 0.2
        
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.5:
            return "high"
        elif risk_score >= 0.2:
            return "medium"
        else:
            return "low"

# Импортируем cv2 для обработки изображений
try:
    import cv2
except ImportError:
    logger.warning("OpenCV not available, thermal image analysis will be limited")
    cv2 = None

if __name__ == "__main__":
    # Тестирование интеграции с тепловыми данными
    thermal_integration = ThermalIntegration()
    
    # Тестовая локация (Москва)
    location = (55.7558, 37.6173)
    
    print("Testing Thermal Integration...")
    
    try:
        thermal_data = thermal_integration.get_thermal_data(location)
        
        print(f"Thermal Data Summary:")
        print(f"  Data Available: {thermal_data.data_available}")
        print(f"  Thermal Anomaly Score: {thermal_data.thermal_anomaly_score:.3f}")
        print(f"  Fire Detection Score: {thermal_data.fire_detection_score:.3f}")
        print(f"  Surface Temperature: {thermal_data.surface_temperature:.1f}°C")
        print(f"  Hotspot Count: {len(thermal_data.hotspot_locations)}")
        
        summary = thermal_integration.get_thermal_summary(thermal_data)
        print(f"\nRisk Level: {summary['risk_level']}")
        
        # Тест проверки активности в зонах
        test_vehicle_locations = [(100, 100), (200, 200)]
        activity_in_zones = thermal_integration.check_thermal_activity_in_zones(
            thermal_data, test_vehicle_locations
        )
        
        print(f"Thermal Activity in Vehicle Zones: {activity_in_zones}")
        
    except Exception as e:
        print(f"Error during thermal integration test: {e}")
        logger.exception("Thermal integration test failed")
