"""
Laptop Only - Упрощенный клиент только для локальной AI обработки
Без лишних функций, только отправка в облако
"""

import cv2
import numpy as np
import time
import asyncio
import websockets
import json
import base64
from typing import Dict

# Core модули
from core.detector import Detector
from core.tracker import Tracker
from core.behavior import BehaviorAnalyzer
from core.thermal import ThermalIntegration
from core.risk import RiskEngine

class LaptopOnlyClient:
    """Максимально простой клиент для ноутбука"""
    
    def __init__(self):
        # URL облачного API
        self.cloud_url = "wss://SafeQunar.onrender.com/ws/laptop"
        
        # AI модули
        self.detector = Detector()
        self.tracker = Tracker(max_disappeared=50)
        self.behavior_analyzer = BehaviorAnalyzer()
        self.thermal_integration = ThermalIntegration()
        self.risk_engine = RiskEngine()
        
        # Статистика
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        
        # Risk сглаживание
        self.prev_risk = 0.0
        self.risk_smoothing = 0.8
        
        # WebSocket
        self.websocket = None
        self.is_connected = False
        
        # Текущие данные
        self.current_data = {
            'frame_id': 0,
            'timestamp': 0,
            'video_frame': '',
            'risk_score': 0.0,
            'risk_level': 'NORMAL',
            'scene': {
                'vehicles': 0,
                'persons': 0,
                'avg_speed': 0.0,
                'convoy_detected': False,
                'thermal_active': False
            },
            'fps': 0.0
        }
    
    async def connect(self):
        """Подключение к облаку"""
        while True:
            try:
                print(f"🔌 Connecting to {self.cloud_url}")
                self.websocket = await websockets.connect(self.cloud_url)
                self.is_connected = True
                print("✅ Connected to SafeQunar Cloud")
                
                # Поддерживаем соединение
                while self.is_connected:
                    try:
                        await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
                    except websockets.exceptions.ConnectionClosed:
                        break
                        
            except Exception as e:
                print(f"❌ Connection error: {e}")
                self.is_connected = False
                
            print("🔄 Reconnecting in 3 seconds...")
            await asyncio.sleep(3)
    
    async def send_data(self, data: Dict):
        """Отправка данных"""
        if self.is_connected and self.websocket:
            try:
                await self.websocket.send(json.dumps(data))
            except:
                self.is_connected = False
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Обработка кадра"""
        
        # Обрабатываем каждый 2-й кадр
        if self.frame_count % 2 != 0:
            return self.current_data
        
        try:
            # Upsample для детекции
            frame_small = cv2.resize(frame, (160, 120))
            frame_upsampled = cv2.resize(frame_small, (320, 320))
            
            # Детекция
            detections = self.detector.detect(frame_upsampled)
            
            # Трекинг
            tracks = self.tracker.update_tracks(detections)
            
            # Анализ поведения
            behavior_features = self.behavior_analyzer.analyze(tracks)
            
            # Тепловые данные
            if not hasattr(self, '_thermal_data'):
                self._thermal_data = self.thermal_integration.get_thermal_data((55.7558, 37.6173))
            
            # Risk Engine
            from core.fusion import FusedFeatures
            fused_features = FusedFeatures(
                convoy_flag=behavior_features.convoy_flag,
                convoy_size=behavior_features.convoy_size,
                anomaly_speed=behavior_features.anomaly_speed,
                anomaly_direction=behavior_features.anomaly_direction,
                density=behavior_features.density,
                thermal_activity=self._thermal_data.thermal_anomaly_score > 0.5,
                thermal_data_available=self._thermal_data.data_available,
                timestamp=time.time()
            )
            
            risk_assessment = self.risk_engine.calculate_risk(fused_features)
            
            # Сглаживание риска
            current_risk = risk_assessment.risk_score
            smoothed_risk = (self.risk_smoothing * self.prev_risk + 
                           (1 - self.risk_smoothing) * current_risk)
            self.prev_risk = smoothed_risk
            
            # Уровень риска
            if smoothed_risk > 0.6:
                risk_level = "ALERT"
            elif smoothed_risk > 0.3:
                risk_level = "MONITOR"
            else:
                risk_level = "NORMAL"
            
            # Статистика
            vehicles = len([t for t in tracks if t.class_name in ['car', 'truck', 'bus']])
            persons = len([t for t in tracks if t.class_name == 'person'])
            
            # Кодируем кадр
            frame_with_overlay = self._add_overlay(frame, tracks, smoothed_risk, risk_level)
            frame_base64 = self._encode_frame(frame_with_overlay)
            
            # Обновляем данные
            self.current_data = {
                'frame_id': self.frame_count,
                'timestamp': time.time(),
                'video_frame': frame_base64,
                'risk_score': smoothed_risk,
                'risk_level': risk_level,
                'scene': {
                    'vehicles': vehicles,
                    'persons': persons,
                    'avg_speed': behavior_features.avg_scene_speed,
                    'convoy_detected': behavior_features.convoy_flag,
                    'thermal_active': fused_features.thermal_activity
                },
                'fps': self.fps
            }
            
            return self.current_data
            
        except Exception as e:
            print(f"Processing error: {e}")
            return self.current_data
    
    def _add_overlay(self, frame: np.ndarray, tracks, risk_score, risk_level) -> np.ndarray:
        """Добавление overlay"""
        overlay = frame.copy()
        
        # Bounding boxes
        for track in tracks:
            if track.disappeared_count == 0:
                x1, y1, x2, y2 = track.bbox
                color = (0, 255, 0) if track.class_name == 'person' else (0, 0, 255)
                
                # Масштабируем к QQVGA
                scale_x = 160 / 320
                scale_y = 120 / 320
                
                x1_scaled = int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                
                cv2.rectangle(overlay, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 1)
                cv2.putText(overlay, f"{track.id}", (x1_scaled, y1_scaled - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Risk индикатор
        risk_color = (0, 255, 0) if risk_level == "NORMAL" else \
                    (0, 255, 255) if risk_level == "MONITOR" else (0, 0, 255)
        
        cv2.rectangle(overlay, (5, 5), (int(155 * risk_score), 15), risk_color, -1)
        cv2.putText(overlay, f"Risk: {risk_score:.2f}", (5, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # FPS
        cv2.putText(overlay, f"FPS: {self.fps:.1f}", (5, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return overlay
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Кодирование кадра"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 12])
        return base64.b64encode(buffer).decode('utf-8')
    
    def update_fps(self):
        """Обновление FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.last_fps_time = current_time
    
    async def run(self, camera_id=0):
        """Основной цикл"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return
        
        # Настройки камеры
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
        cap.set(cv2.CAP_PROP_FPS, 10)
        
        print("📹 Starting SafeQunar AI Client...")
        print("🌐 Dashboard: https://SafeQunar.onrender.com/")
        print("📡 Sending to: wss://SafeQunar.onrender.com/ws/laptop")
        print("⏹️  Press 'q' to quit")
        print()
        
        # Запуск подключения в фоне
        connection_task = asyncio.create_task(self.connect())
        
        # Основной цикл
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            self.update_fps()
            
            # Обработка кадра
            data = self.process_frame(frame)
            
            # Отправка в облако
            await self.send_data(data)
            
            # Локальный показ
            cv2.imshow('SafeQunar AI Client', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Очистка
        cap.release()
        cv2.destroyAllWindows()
        if self.websocket:
            await self.websocket.close()

async def main():
    """Запуск"""
    print("🛡️  SafeQunar Laptop Client")
    print("=" * 40)
    
    client = LaptopOnlyClient()
    
    try:
        await client.run(0)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
