"""
ESP32-CAM Stream Receiver and Processor
Для приема MJPEG потока с ESP32-CAM и обработки в реальном времени
"""

import cv2
import numpy as np
import requests
import threading
import time
import queue
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ESP32Config:
    """Конфигурация ESP32-CAM"""
    ip_address: str
    port: int = 81
    stream_path: str = "/stream"
    timeout: int = 10
    retry_interval: float = 5.0
    max_retries: int = 3

@dataclass
class SensorReading:
    """Данные с сенсоров ESP32"""
    timestamp: float
    temperature: float
    humidity: float
    light_level: float
    battery_level: float
    signal_strength: float
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None

class ESP32StreamReceiver:
    """Приемник MJPEG потока с ESP32-CAM"""
    
    def __init__(self, config: ESP32Config):
        self.config = config
        self.stream_url = f"http://{config.ip_address}:{config.port}{config.stream_path}"
        self.sensor_url = f"http://{config.ip_address}:{config.port}/sensor"
        
        self.is_running = False
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.sensor_queue = queue.Queue(maxsize=100)
        
        # Статистика
        self.frame_count = 0
        self.last_frame_time = 0
        self.fps = 0.0
        self.connection_errors = 0
        
        # OpenCV capture
        self.cap = None
        
    def start_stream(self) -> bool:
        """Запуск приема потока"""
        try:
            # Инициализация OpenCV VideoCapture для MJPEG потока
            self.cap = cv2.VideoCapture(self.stream_url)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open stream: {self.stream_url}")
                return False
            
            # Настройки потока
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            # Запуск сбора данных с сенсоров
            sensor_thread = threading.Thread(target=self._sensor_loop)
            sensor_thread.daemon = True
            sensor_thread.start()
            
            logger.info(f"ESP32 stream started: {self.stream_url}")
            return True
            
        except Exception as esp32_error:
            logger.error(f"Error starting ESP32 stream: {esp32_error}")
            return False
    
    def stop_stream(self):
        """Остановка приема потока"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("ESP32 stream stopped")
    
    def _capture_loop(self):
        """Основной цикл захвата кадров"""
        retry_count = 0
        
        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logger.warning("Stream connection lost, attempting to reconnect...")
                    if self._reconnect_stream():
                        retry_count = 0
                    else:
                        retry_count += 1
                        if retry_count >= self.config.max_retries:
                            logger.error("Max retries reached, stopping stream")
                            break
                        time.sleep(self.config.retry_interval)
                    continue
                
                # Захват кадра
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    self.frame_count += 1
                    current_time = time.time()
                    
                    # Расчет FPS
                    if self.last_frame_time > 0:
                        frame_time = current_time - self.last_frame_time
                        self.fps = 1.0 / frame_time if frame_time > 0 else 0
                    
                    self.last_frame_time = current_time
                    
                    # Добавляем в очередь
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                    else:
                        # Если очередь полна, удаляем старый кадр
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(frame)
                        except queue.Empty:
                            pass
                    
                    retry_count = 0  # Сброс счетчика ошибок при успешном захвате
                    
                else:
                    logger.warning("Failed to capture frame")
                    retry_count += 1
                    if retry_count >= self.config.max_retries:
                        logger.error("Too many capture failures, reconnecting...")
                        self._reconnect_stream()
                        retry_count = 0
                
                # Небольшая задержка для управления нагрузкой
                time.sleep(0.01)
                
            except Exception as esp32_error:
                logger.error(f"Error in capture loop: {esp32_error}")
                self.connection_errors += 1
                retry_count += 1
                
                if retry_count >= self.config.max_retries:
                    logger.error("Too many errors, attempting reconnect...")
                    self._reconnect_stream()
                    retry_count = 0
                
                time.sleep(self.config.retry_interval)
    
    def _sensor_loop(self):
        """Цикл сбора данных с сенсоров"""
        while self.is_running:
            try:
                # Запрос данных с сенсоров
                sensor_data = self._fetch_sensor_data()
                
                if sensor_data:
                    if not self.sensor_queue.full():
                        self.sensor_queue.put(sensor_data)
                
                time.sleep(1.0)  # Обновление сенсоров каждую секунду
                
            except Exception as esp32_error:
                logger.error(f"Error fetching sensor data: {esp32_error}")
                time.sleep(5.0)
    
    def _fetch_sensor_data(self) -> Optional[SensorReading]:
        """Получение данных с сенсоров"""
        try:
            response = requests.get(
                self.sensor_url,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                return SensorReading(
                    timestamp=time.time(),
                    temperature=data.get('temperature', 0.0),
                    humidity=data.get('humidity', 0.0),
                    light_level=data.get('light', 0.0),
                    battery_level=data.get('battery', 100.0),
                    signal_strength=data.get('wifi_rssi', -50.0),
                    gps_lat=data.get('gps', {}).get('lat'),
                    gps_lon=data.get('gps', {}).get('lon')
                )
            else:
                logger.warning(f"Sensor request failed: {response.status_code}")
                return None
                
        except Exception as esp32_error:
            logger.error(f"Error fetching sensor data: {esp32_error}")
            return None
    
    def _reconnect_stream(self) -> bool:
        """Переподключение к потоку"""
        try:
            if self.cap:
                self.cap.release()
            
            time.sleep(1.0)  # Небольшая пауза перед переподключением
            
            self.cap = cv2.VideoCapture(self.stream_url)
            
            if self.cap.isOpened():
                logger.info("Successfully reconnected to stream")
                return True
            else:
                logger.error("Failed to reconnect to stream")
                return False
                
        except Exception as esp32_error:
            logger.error(f"Error during reconnection: {esp32_error}")
            return False
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Получение кадра из очереди"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_sensor_data(self, timeout: float = 1.0) -> Optional[SensorReading]:
        """Получение данных с сенсоров"""
        try:
            return self.sensor_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики"""
        return {
            'is_running': self.is_running,
            'frame_count': self.frame_count,
            'fps': self.fps,
            'connection_errors': self.connection_errors,
            'queue_sizes': {
                'frames': self.frame_queue.qsize(),
                'sensors': self.sensor_queue.qsize()
            },
            'stream_url': self.stream_url,
            'last_frame_time': self.last_frame_time
        }

class ESP32Simulator:
    """Симулятор ESP32-CAM для тестирования"""
    
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.is_running = False
        self.frame_count = 0
        
    def start_simulation(self) -> bool:
        """Запуск симуляции"""
        self.is_running = True
        logger.info("ESP32 simulation started")
        return True
    
    def stop_simulation(self):
        """Остановка симуляции"""
        self.is_running = False
        logger.info("ESP32 simulation stopped")
    
    def generate_frame(self) -> np.ndarray:
        """Генерация тестового кадра"""
        self.frame_count += 1
        
        # Создаем тестовый кадр
        frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        
        # Добавляем движущиеся объекты для теста детекции
        center_x = int(self.width / 2 + 100 * np.sin(self.frame_count * 0.05))
        center_y = int(self.height / 2 + 50 * np.cos(self.frame_count * 0.05))
        
        cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), -1)
        cv2.rectangle(frame, (center_x - 50, center_y - 50), 
                     (center_x + 50, center_y + 50), (255, 0, 0), 2)
        
        # Добавляем текст
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def generate_sensor_data(self) -> SensorReading:
        """Генерация тестовых данных сенсоров"""
        return SensorReading(
            timestamp=time.time(),
            temperature=20 + 5 * np.sin(self.frame_count * 0.01),
            humidity=50 + 10 * np.cos(self.frame_count * 0.01),
            light_level=500 + 200 * np.sin(self.frame_count * 0.02),
            battery_level=max(20, 100 - self.frame_count * 0.01),
            signal_strength=-50 - 10 * np.random.random(),
            gps_lat=55.7558 + 0.001 * np.random.random(),
            gps_lon=37.6173 + 0.001 * np.random.random()
        )

class EdgeDeviceManager:
    """Менеджер edge устройств"""
    
    def __init__(self):
        self.devices = {}
        self.active_device = None
        
    def add_esp32_device(self, device_id: str, config: ESP32Config) -> bool:
        """Добавление ESP32 устройства"""
        try:
            receiver = ESP32StreamReceiver(config)
            self.devices[device_id] = receiver
            logger.info(f"ESP32 device {device_id} added")
            return True
        except Exception as esp32_error:
            logger.error(f"Failed to add ESP32 device: {esp32_error}")
            return False
    
    def add_simulator(self, device_id: str) -> bool:
        """Добавление симулятора"""
        try:
            simulator = ESP32Simulator()
            self.devices[device_id] = simulator
            logger.info(f"Simulator device {device_id} added")
            return True
        except Exception as esp32_error:
            logger.error(f"Failed to add simulator: {esp32_error}")
            return False
    
    def start_device(self, device_id: str) -> bool:
        """Запуск устройства"""
        if device_id not in self.devices:
            logger.error(f"Device {device_id} not found")
            return False
        
        device = self.devices[device_id]
        
        if isinstance(device, ESP32StreamReceiver):
            success = device.start_stream()
        elif isinstance(device, ESP32Simulator):
            success = device.start_simulation()
        else:
            success = False
        
        if success:
            self.active_device = device_id
            logger.info(f"Device {device_id} started successfully")
        
        return success
    
    def stop_device(self, device_id: str) -> bool:
        """Остановка устройства"""
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        
        if isinstance(device, ESP32StreamReceiver):
            device.stop_stream()
        elif isinstance(device, ESP32Simulator):
            device.stop_simulation()
        
        if self.active_device == device_id:
            self.active_device = None
        
        logger.info(f"Device {device_id} stopped")
        return True
    
    def get_frame(self, device_id: str = None) -> Optional[np.ndarray]:
        """Получение кадра от устройства"""
        if device_id is None:
            device_id = self.active_device
        
        if device_id is None or device_id not in self.devices:
            return None
        
        device = self.devices[device_id]
        
        if isinstance(device, ESP32StreamReceiver):
            return device.get_frame()
        elif isinstance(device, ESP32Simulator):
            return device.generate_frame()
        
        return None
    
    def get_sensor_data(self, device_id: str = None) -> Optional[SensorReading]:
        """Получение данных сенсоров от устройства"""
        if device_id is None:
            device_id = self.active_device
        
        if device_id is None or device_id not in self.devices:
            return None
        
        device = self.devices[device_id]
        
        if isinstance(device, ESP32StreamReceiver):
            return device.get_sensor_data()
        elif isinstance(device, ESP32Simulator):
            return device.generate_sensor_data()
        
        return None
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Получение статистики всех устройств"""
        stats = {
            'total_devices': len(self.devices),
            'active_device': self.active_device,
            'devices': {}
        }
        
        for device_id, device in self.devices.items():
            if isinstance(device, ESP32StreamReceiver):
                stats['devices'][device_id] = device.get_statistics()
            elif isinstance(device, ESP32Simulator):
                stats['devices'][device_id] = {
                    'is_running': device.is_running,
                    'frame_count': device.frame_count,
                    'type': 'simulator'
                }
        
        return stats

# Тестирование
if __name__ == "__main__":
    print("Testing ESP32-CAM Edge Layer...")
    
    # Создаем менеджер устройств
    device_manager = EdgeDeviceManager()
    
    # Добавляем симулятор для тестирования
    device_manager.add_simulator("test_simulator")
    
    try:
        # Запускаем устройство
        if device_manager.start_device("test_simulator"):
            print("Device started successfully")
            
            # Тестовый цикл
            for i in range(100):
                frame = device_manager.get_frame()
                sensor_data = device_manager.get_sensor_data()
                
                if frame is not None:
                    print(f"Frame {i}: shape={frame.shape}")
                
                if sensor_data is not None:
                    print(f"Sensor data: temp={sensor_data.temperature:.1f}°C, "
                          f"humidity={sensor_data.humidity:.1f}%")
                
                time.sleep(0.1)
            
            # Статистика
            stats = device_manager.get_all_statistics()
            print(f"\nStatistics: {json.dumps(stats, indent=2, default=str)}")
            
        else:
            print("Failed to start device")
            
    except Exception as esp32_error:
        print(f"Error during testing: {esp32_error}")
    
    finally:
        # Остановка
        device_manager.stop_device("test_simulator")
        print("Test completed")
