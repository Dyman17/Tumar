import requests
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import time
from urllib.parse import urlencode
import cv2
from PIL import Image
import io
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NASAGIBSClient:
    """Клиент для NASA GIBS (Global Imagery Browse Services)"""
    
    def __init__(self):
        self.base_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
        self.tile_url = "https://gibs.earthdata.nasa.gov/twms/epsg4326/best/twms.cgi"
        
        # Доступные слои для мониторинга
        self.layers = {
            # Тепловые аномалии
            'MODIS_Terra_Thermal_Anomalies_Day': {
                'name': 'thermal_anomalies_day',
                'description': 'MODIS Terra Thermal Anomalies - Day',
                'resolution': '1000m',
                'temporal_resolution': 'daily'
            },
            'MODIS_Aqua_Thermal_Anomalies_Day': {
                'name': 'thermal_anomalies_day_aqua',
                'description': 'MODIS Aqua Thermal Anomalies - Day',
                'resolution': '1000m',
                'temporal_resolution': 'daily'
            },
            
            # Детекция огня
            'MODIS_Terra_Fire_Detection_Day': {
                'name': 'fire_detection_day',
                'description': 'MODIS Terra Fire Detection - Day',
                'resolution': '1000m',
                'temporal_resolution': 'daily'
            },
            'MODIS_Aqua_Fire_Detection_Day': {
                'name': 'fire_detection_day_aqua',
                'description': 'MODIS Aqua Fire Detection - Day',
                'resolution': '1000m',
                'temporal_resolution': 'daily'
            },
            
            # NDVI (индекс растительности)
            'MODIS_Terra_VEGETATION_Index_Daily': {
                'name': 'ndvi_daily',
                'description': 'MODIS Terra NDVI - Daily',
                'resolution': '250m',
                'temporal_resolution': 'daily'
            },
            
            # Поверхностная температура
            'MODIS_Terra_LST_Day': {
                'name': 'surface_temperature_day',
                'description': 'MODIS Terra Land Surface Temperature - Day',
                'resolution': '1000m',
                'temporal_resolution': 'daily'
            },
            
            # Истинный цвет
            'MODIS_Terra_TrueColor': {
                'name': 'true_color',
                'description': 'MODIS Terra True Color',
                'resolution': '250m',
                'temporal_resolution': 'daily'
            }
        }
        
        # Кэш для запросов
        self.cache = {}
        self.cache_timeout = 3600  # 1 час
        
    def get_available_layers(self) -> Dict:
        """Получение списка доступных слоев"""
        return self.layers
    
    def build_wms_url(self, layer: str, bbox: Tuple[float, float, float, float], 
                     width: int = 512, height: int = 512, 
                     time_str: str = None, format: str = 'image/png') -> str:
        """Построение WMS URL"""
        
        if layer not in self.layers:
            raise ValueError(f"Layer {layer} not available")
        
        # Если время не указано, используем текущую дату
        if time_str is None:
            time_str = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'SERVICE': 'WMS',
            'VERSION': '1.3.0',
            'REQUEST': 'GetMap',
            'LAYERS': layer,
            'FORMAT': format,
            'TRANSPARENT': 'TRUE',
            'BBOX': f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",  # WMS 1.3.0 порядок
            'WIDTH': width,
            'HEIGHT': height,
            'TIME': time_str
        }
        
        return f"{self.base_url}?{urlencode(params)}"
    
    def build_tile_url(self, layer: str, tile_matrix: str, 
                      tile_row: int, tile_col: int, 
                      time_str: str = None) -> str:
        """Построение URL для тайла"""
        
        if layer not in self.layers:
            raise ValueError(f"Layer {layer} not available")
        
        if time_str is None:
            time_str = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'SERVICE': 'WMTS',
            'VERSION': '1.0.0',
            'REQUEST': 'GetTile',
            'LAYER': layer,
            'STYLE': 'default',
            'FORMAT': 'image/png',
            'TILEMATRIXSET': tile_matrix,
            'TILEMATRIX': tile_matrix,
            'TILEROW': str(tile_row),
            'TILECOL': str(tile_col),
            'TIME': time_str
        }
        
        return f"{self.tile_url}?{urlencode(params)}"
    
    def get_thermal_map(self, bbox: Tuple[float, float, float, float], 
                       date: datetime = None, width: int = 512, 
                       height: int = 512) -> np.ndarray:
        """Получение тепловой карты"""
        
        if date is None:
            date = datetime.now()
        
        time_str = date.strftime('%Y-%m-%d')
        
        # Пробуем разные слои тепловых аномалий
        thermal_layers = [
            'MODIS_Terra_Thermal_Anomalies_Day',
            'MODIS_Aqua_Thermal_Anomalies_Day'
        ]
        
        for layer in thermal_layers:
            try:
                url = self.build_wms_url(layer, bbox, width, height, time_str)
                image_data = self.fetch_image(url)
                
                if image_data is not None:
                    logger.info(f"Successfully fetched thermal data from {layer}")
                    return image_data
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {layer}: {e}")
                continue
        
        logger.warning("No thermal data available")
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    def get_fire_map(self, bbox: Tuple[float, float, float, float], 
                    date: datetime = None, width: int = 512, 
                    height: int = 512) -> np.ndarray:
        """Получение карты пожаров"""
        
        if date is None:
            date = datetime.now()
        
        time_str = date.strftime('%Y-%m-%d')
        
        # Пробуем слои детекции огня
        fire_layers = [
            'MODIS_Terra_Fire_Detection_Day',
            'MODIS_Aqua_Fire_Detection_Day'
        ]
        
        for layer in fire_layers:
            try:
                url = self.build_wms_url(layer, bbox, width, height, time_str)
                image_data = self.fetch_image(url)
                
                if image_data is not None:
                    logger.info(f"Successfully fetched fire data from {layer}")
                    return image_data
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {layer}: {e}")
                continue
        
        logger.warning("No fire data available")
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    def get_ndvi_map(self, bbox: Tuple[float, float, float, float], 
                    date: datetime = None, width: int = 512, 
                    height: int = 512) -> np.ndarray:
        """Получение NDVI карты"""
        
        if date is None:
            date = datetime.now()
        
        time_str = date.strftime('%Y-%m-%d')
        
        try:
            url = self.build_wms_url('MODIS_Terra_VEGETATION_Index_Daily', 
                                   bbox, width, height, time_str)
            image_data = self.fetch_image(url)
            
            if image_data is not None:
                logger.info("Successfully fetched NDVI data")
                return image_data
                
        except Exception as e:
            logger.warning(f"Failed to fetch NDVI data: {e}")
        
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    def get_surface_temperature(self, bbox: Tuple[float, float, float, float], 
                               date: datetime = None, width: int = 512, 
                               height: int = 512) -> np.ndarray:
        """Получение карты поверхностной температуры"""
        
        if date is None:
            date = datetime.now()
        
        time_str = date.strftime('%Y-%m-%d')
        
        try:
            url = self.build_wms_url('MODIS_Terra_LST_Day', 
                                   bbox, width, height, time_str)
            image_data = self.fetch_image(url)
            
            if image_data is not None:
                logger.info("Successfully fetched surface temperature data")
                return image_data
                
        except Exception as e:
            logger.warning(f"Failed to fetch surface temperature data: {e}")
        
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    def fetch_image(self, url: str) -> Optional[np.ndarray]:
        """Загрузка изображения по URL"""
        
        # Проверяем кэш
        cache_key = hash(url)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Конвертируем в numpy array
            image = Image.open(io.BytesIO(response.content))
            image_array = np.array(image)
            
            # Если изображение в RGBA, конвертируем в RGB
            if image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            
            # Кэшируем результат
            self.cache[cache_key] = (image_array, time.time())
            
            return image_array
            
        except Exception as e:
            logger.error(f"Failed to fetch image from {url}: {e}")
            return None
    
    def analyze_thermal_anomalies(self, thermal_data: np.ndarray) -> Dict:
        """Анализ тепловых аномалий"""
        
        if thermal_data.size == 0:
            return {'anomaly_score': 0.0, 'hotspots': [], 'statistics': {}}
        
        # Конвертируем в градации серого если нужно
        if len(thermal_data.shape) == 3:
            gray = cv2.cvtColor(thermal_data, cv2.COLOR_RGB2GRAY)
        else:
            gray = thermal_data
        
        # Статистика
        mean_temp = np.mean(gray)
        std_temp = np.std(gray)
        max_temp = np.max(gray)
        
        # Поиск горячих точек (выше среднего + 2 ст. отклонения)
        threshold = mean_temp + 2 * std_temp
        hotspots = []
        
        # Находим контуры горячих областей
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # Минимальный размер
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                intensity = np.mean(gray[y:y+h, x:x+w])
                
                hotspots.append({
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'intensity': float(intensity),
                    'area': float(cv2.contourArea(contour))
                })
        
        # Оценка аномалии
        anomaly_score = min(max_temp / 255.0, 1.0)  # Нормализованный score
        
        return {
            'anomaly_score': anomaly_score,
            'hotspots': hotspots,
            'statistics': {
                'mean_temperature': float(mean_temp),
                'std_temperature': float(std_temp),
                'max_temperature': float(max_temp),
                'hotspot_count': len(hotspots)
            }
        }
    
    def get_multi_layer_data(self, bbox: Tuple[float, float, float, float], 
                           date: datetime = None, width: int = 512, 
                           height: int = 512) -> Dict:
        """Получение данных с нескольких слоев"""
        
        if date is None:
            date = datetime.now()
        
        results = {}
        
        # Тепловые аномалии
        thermal_data = self.get_thermal_map(bbox, date, width, height)
        if thermal_data.size > 0:
            results['thermal'] = self.analyze_thermal_anomalies(thermal_data)
            results['thermal_image'] = thermal_data
        
        # Детекция огня
        fire_data = self.get_fire_map(bbox, date, width, height)
        if fire_data.size > 0:
            results['fire'] = self.analyze_thermal_anomalies(fire_data)
            results['fire_image'] = fire_data
        
        # NDVI
        ndvi_data = self.get_ndvi_map(bbox, date, width, height)
        if ndvi_data.size > 0:
            results['ndvi_image'] = ndvi_data
            # Анализ NDVI
            if len(ndvi_data.shape) == 3:
                ndvi_gray = cv2.cvtColor(ndvi_data, cv2.COLOR_RGB2GRAY)
            else:
                ndvi_gray = ndvi_data
            
            results['ndvi'] = {
                'mean_ndvi': float(np.mean(ndvi_gray)),
                'vegetation_health': 'good' if np.mean(ndvi_gray) > 128 else 'poor'
            }
        
        # Поверхностная температура
        temp_data = self.get_surface_temperature(bbox, date, width, height)
        if temp_data.size > 0:
            results['temperature_image'] = temp_data
            temp_analysis = self.analyze_thermal_anomalies(temp_data)
            results['surface_temperature'] = temp_analysis
        
        results['metadata'] = {
            'bbox': bbox,
            'date': date.isoformat(),
            'resolution': (width, height),
            'layers_available': list(results.keys())
        }
        
        return results
    
    def create_overlay_image(self, base_image: np.ndarray, 
                           overlay_data: Dict, alpha: float = 0.6) -> np.ndarray:
        """Создание overlay изображения"""
        
        if base_image is None or base_image.size == 0:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        
        overlay = base_image.copy()
        
        # Накладываем тепловые данные
        if 'thermal_image' in overlay_data:
            thermal = overlay_data['thermal_image']
            if thermal.shape[:2] == base_image.shape[:2]:
                # Создаем цветовую карту для тепловых данных
                thermal_colored = cv2.applyColorMap(
                    cv2.cvtColor(thermal, cv2.COLOR_RGB2GRAY), 
                    cv2.COLORMAP_JET
                )
                overlay = cv2.addWeighted(overlay, 1-alpha, thermal_colored, alpha, 0)
        
        # Накладываем данные о пожарах
        if 'fire_image' in overlay_data:
            fire = overlay_data['fire_image']
            if fire.shape[:2] == base_image.shape[:2]:
                # Выделяем огонь красным
                fire_mask = np.sum(fire, axis=2) > 128
                fire_colored = np.zeros_like(fire)
                fire_colored[fire_mask] = [0, 0, 255]  # Красный в BGR
                overlay = cv2.addWeighted(overlay, 1, fire_colored, 0.8, 0)
        
        return overlay

class ThermalAnomalyAnalyzer:
    """Анализатор тепловых аномалий"""
    
    def __init__(self, nasa_client: NASAGIBSClient):
        self.nasa_client = nasa_client
        self.anomaly_history = []
        
    def analyze_location(self, lat: float, lon: float, radius: float = 0.1) -> Dict:
        """Анализ локации на тепловые аномалии"""
        
        # Формируем bbox
        bbox = (lon - radius, lat - radius, lon + radius, lat + radius)
        
        # Получаем данные
        data = self.nasa_client.get_multi_layer_data(bbox)
        
        # Комплексный анализ
        risk_score = 0.0
        risk_factors = []
        
        # Анализ тепловых аномалий
        if 'thermal' in data:
            thermal_score = data['thermal']['anomaly_score']
            risk_score += thermal_score * 0.4
            risk_factors.append(f"Thermal anomalies: {thermal_score:.2f}")
        
        # Анализ пожаров
        if 'fire' in data:
            fire_score = data['fire']['anomaly_score']
            risk_score += fire_score * 0.5
            risk_factors.append(f"Fire detection: {fire_score:.2f}")
        
        # Анализ температуры поверхности
        if 'surface_temperature' in data:
            temp_score = data['surface_temperature']['anomaly_score']
            risk_score += temp_score * 0.3
            risk_factors.append(f"Surface temperature: {temp_score:.2f}")
        
        # Нормализуем score
        risk_score = min(risk_score, 1.0)
        
        # Классификация риска
        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.6:
            risk_level = "MEDIUM"
        elif risk_score < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        result = {
            'location': {'lat': lat, 'lon': lon},
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'analysis_timestamp': datetime.now().isoformat(),
            'data_layers': list(data.keys()),
            'detailed_analysis': data
        }
        
        # Сохраняем в историю
        self.anomaly_history.append(result)
        
        return result

if __name__ == "__main__":
    # Тестирование NASA GIBS клиента
    nasa_client = NASAGIBSClient()
    analyzer = ThermalAnomalyAnalyzer(nasa_client)
    
    # Тестовые координаты (Москва)
    lat, lon = 55.7558, 37.6173
    
    print("Testing NASA GIBS integration...")
    
    try:
        # Анализ локации
        result = analyzer.analyze_location(lat, lon, radius=0.05)
        
        print(f"Analysis result for ({lat}, {lon}):")
        print(f"Risk level: {result['risk_level']}")
        print(f"Risk score: {result['risk_score']:.3f}")
        print(f"Risk factors: {result['risk_factors']}")
        
        # Сохраняем результат
        with open('thermal_analysis_result.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        logger.exception("Analysis failed")
