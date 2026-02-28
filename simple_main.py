"""
Simple Main - Максимально стабильный пайплайн без логов
QQVGA (160×120) → 5 FPS AI → Real-time WebSocket
"""

import cv2
import numpy as np
import time
import threading
import asyncio
import websockets
import json
import base64
from typing import Dict, List
import io

# Core модули
from core.detector import Detector
from core.tracker import Tracker
from core.behavior import BehaviorAnalyzer
from core.thermal import ThermalIntegration
from core.risk import RiskEngine

class SimpleMonitoringSystem:
    """Максимально простая система мониторинга"""
    
    def __init__(self, location=(55.7558, 37.6173)):
        # AI модули
        self.detector = Detector()
        self.tracker = Tracker(max_disappeared=50)  # Увеличен для QQVGA
        self.behavior_analyzer = BehaviorAnalyzer()
        self.thermal_integration = ThermalIntegration()
        self.risk_engine = RiskEngine()
        
        # Локация
        self.location = location
        
        # Статистика
        self.frame_count = 0
        self.processed_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        
        # Risk сглаживание
        self.prev_risk = 0.0
        self.risk_smoothing = 0.8
        
        # WebSocket клиенты
        self.websocket_clients = set()
        
        # Флаг работы
        self.is_running = False
        
        # Текущие данные для веб-интерфейса
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
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Обработка кадра (каждый 2-й)"""
        
        start_time = time.time()
        
        # Обрабатываем только каждый 2-й кадр для стабильности
        if self.frame_count % 2 != 0:
            return self.current_data
        
        try:
            # 1. Upsample для лучшей детекции
            frame_small = cv2.resize(frame, (160, 120))
            frame_upsampled = cv2.resize(frame_small, (320, 320))
            
            # 2. Детекция
            detections = self.detector.detect(frame_upsampled)
            
            # 3. Трекинг
            tracks = self.tracker.update_tracks(detections)
            
            # 4. Анализ поведения
            behavior_features = self.behavior_analyzer.analyze(tracks)
            
            # 5. Тепловые данные (кэшируем)
            if not hasattr(self, '_thermal_data'):
                self._thermal_data = self.thermal_integration.get_thermal_data(self.location)
            
            # 6. Risk Engine
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
            
            # 7. Сглаживание риска
            current_risk = risk_assessment.risk_score
            smoothed_risk = (self.risk_smoothing * self.prev_risk + 
                           (1 - self.risk_smoothing) * current_risk)
            self.prev_risk = smoothed_risk
            
            # 8. Определение уровня
            if smoothed_risk > 0.6:
                risk_level = "ALERT"
            elif smoothed_risk > 0.3:
                risk_level = "MONITOR"
            else:
                risk_level = "NORMAL"
            
            # 9. Сбор статистики сцены
            vehicles = len([t for t in tracks if t.class_name in ['car', 'truck', 'bus']])
            persons = len([t for t in tracks if t.class_name == 'person'])
            
            # 10. Обновляем текущие данные
            self.current_data = {
                'frame_id': self.frame_count,
                'timestamp': time.time(),
                'video_frame': self._encode_frame_with_overlay(frame, tracks, smoothed_risk, risk_level),
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
            
            self.processed_count += 1
            
            return self.current_data
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return self.current_data
    
    def _encode_frame_with_overlay(self, frame: np.ndarray, tracks, risk_score, risk_level) -> str:
        """Кодирование кадра с overlay"""
        
        # Создаем overlay
        overlay = frame.copy()
        
        # Рисуем bounding boxes
        for track in tracks:
            if track.disappeared_count == 0:
                x1, y1, x2, y2 = track.bbox
                color = (0, 255, 0) if track.class_name == 'person' else (0, 0, 255)
                
                # Масштабируем bbox обратно к QQVGA
                scale_x = 160 / 320
                scale_y = 120 / 320
                
                x1_scaled = int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                
                cv2.rectangle(overlay, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 1)
                
                # ID
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
        
        # Кодируем
        _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 12])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return frame_base64
    
    def update_fps(self):
        """Обновление FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.last_fps_time = current_time
    
    async def websocket_handler(self, websocket, path):
        """WebSocket обработчик"""
        self.websocket_clients.add(websocket)
        print(f"Client connected: {len(self.websocket_clients)} clients")
        
        try:
            while self.is_running:
                # Отправляем текущие данные
                await websocket.send(json.dumps(self.current_data))
                await asyncio.sleep(0.1)  # 10 FPS для WebSocket
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.websocket_clients.discard(websocket)
            print(f"Client disconnected: {len(self.websocket_clients)} clients")
    
    async def broadcast_data(self):
        """Рассылка данных всем клиентам"""
        if self.websocket_clients:
            message = json.dumps(self.current_data)
            disconnected = set()
            
            for client in self.websocket_clients:
                try:
                    await client.send(message)
                except:
                    disconnected.add(client)
            
            self.websocket_clients.difference_update(disconnected)
    
    def run_camera_loop(self, camera_id=0):
        """Основной цикл с камерой"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return
        
        # Устанавливаем QQVGA
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
        cap.set(cv2.CAP_PROP_FPS, 10)
        
        print("Starting simple monitoring system...")
        print("Open browser: http://localhost:8000")
        print("Press 'q' to quit")
        
        self.is_running = True
        
        # WebSocket сервер
        start_server = websockets.serve(self.websocket_handler, "localhost", 8000)
        asyncio.get_event_loop().run_until_complete(start_server)
        
        # Запуск broadcast в отдельном потоке
        def broadcast_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.broadcast_loop())
        
        broadcast_thread = threading.Thread(target=broadcast_loop)
        broadcast_thread.daemon = True
        broadcast_thread.start()
        
        # Основной цикл
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            self.update_fps()
            
            # Обрабатываем кадр
            self.process_frame(frame)
            
            # Показываем локально (опционально)
            cv2.imshow('Simple AI Monitor', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.is_running = False
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Запуск системы"""
    print("="*50)
    print("SIMPLE AI MONITORING SYSTEM")
    print("QQVGA (160×120) → 5 FPS AI → Real-time WebSocket")
    print("="*50)
    
    system = SimpleMonitoringSystem()
    
    try:
        system.run_camera_loop(0)
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
