"""
Main Pipeline - Основной цикл обработки
INPUT (Video + Thermal Layer) → Detection → Tracking → Behavior → Fusion → Risk → Events → API
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Tuple
import json

# Импортируем все модули core/
from core.detector import Detector
from core.tracker import Tracker
from core.behavior import BehaviorAnalyzer
from core.thermal import ThermalIntegration
from core.fusion import FusionEngine
from core.risk import RiskEngine
from core.events import EventGenerator

class MonitoringPipeline:
    """Основной пайплайн мониторинга"""
    
    def __init__(self, location: Optional[Tuple[float, float]] = None):
        # Инициализация всех модулей
        self.detector = Detector()
        self.tracker = Tracker()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.thermal_integration = ThermalIntegration()
        self.fusion_engine = FusionEngine()
        self.risk_engine = RiskEngine()
        self.event_generator = EventGenerator()
        
        # Локация для тепловых данных
        self.location = location or (55.7558, 37.6173)  # Москва по умолчанию
        
        # Статистика пайплайна
        self.frame_count = 0
        self.processing_times = []
        self.last_thermal_update = 0
        self.thermal_update_interval = 300  # 5 минут
        
        # Флаг работы
        self.is_running = False
        
    def process_frame(self, frame: np.ndarray) -> dict:
        """Обработка одного кадра через весь пайплайн"""
        
        start_time = time.time()
        
        try:
            # 1. Detection Layer
            detections = self.detector.detect(frame)
            
            # 2. Tracking Layer
            tracks = self.tracker.update_tracks(detections)
            
            # 3. Behavior Analysis Layer
            behavior_features = self.behavior_analyzer.analyze(tracks)
            
            # 4. Thermal Integration Layer (обновляется реже)
            current_time = time.time()
            if current_time - self.last_thermal_update > self.thermal_update_interval:
                thermal_data = self.thermal_integration.get_thermal_data(self.location)
                self.last_thermal_update = current_time
            else:
                # Используем кэшированные данные
                if hasattr(self, '_cached_thermal_data'):
                    thermal_data = self._cached_thermal_data
                else:
                    thermal_data = self.thermal_integration.get_thermal_data(self.location)
                    self._cached_thermal_data = thermal_data
                    self.last_thermal_update = current_time
            
            # 5. Fusion Engine
            fused_features = self.fusion_engine.fuse_data(
                tracks, behavior_features, thermal_data, self.location
            )
            
            # 6. Risk Engine
            risk_assessment = self.risk_engine.calculate_risk(fused_features)
            
            # 7. Event Generator
            event = None
            if self.risk_engine.should_generate_event(risk_assessment):
                event = self.event_generator.generate_event(
                    risk_assessment, tracks, fused_features, self.location
                )
            
            # 8. Время обработки
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            self.frame_count += 1
            
            # 9. Формируем результат
            result = {
                'frame_id': self.frame_count,
                'timestamp': current_time,
                'processing_time': processing_time,
                'detections': len(detections),
                'tracks': len(tracks),
                'behavior_features': {
                    'convoy_flag': behavior_features.convoy_flag,
                    'convoy_size': behavior_features.convoy_size,
                    'anomaly_speed': behavior_features.anomaly_speed,
                    'anomaly_direction': behavior_features.anomaly_direction,
                    'density': behavior_features.density,
                    'avg_scene_speed': behavior_features.avg_scene_speed
                },
                'thermal_data': {
                    'available': thermal_data.data_available,
                    'thermal_anomaly_score': thermal_data.thermal_anomaly_score,
                    'fire_detection_score': thermal_data.fire_detection_score,
                    'surface_temperature': thermal_data.surface_temperature,
                    'hotspot_count': len(thermal_data.hotspot_locations)
                },
                'fused_features': {
                    'convoy_flag': fused_features.convoy_flag,
                    'anomaly_speed': fused_features.anomaly_speed,
                    'anomaly_direction': fused_features.anomaly_direction,
                    'density': fused_features.density,
                    'thermal_activity': fused_features.thermal_activity
                },
                'risk_assessment': {
                    'risk_score': risk_assessment.risk_score,
                    'risk_level': risk_assessment.risk_level.value,
                    'primary_factors': risk_assessment.primary_factors
                },
                'event': {
                    'generated': event is not None,
                    'event_id': event.event_id if event else None,
                    'event_type': event.event_type.value if event else None,
                    'severity': event.severity.value if event else None
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            return {
                'frame_id': self.frame_count,
                'timestamp': time.time(),
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def create_visualization(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Создание визуализации с результатами"""
        
        vis_frame = frame.copy()
        
        # Получаем треки для визуализации
        tracks = self.tracker.tracks.values()
        active_tracks = [t for t in tracks if t.disappeared_count == 0]
        
        # 1. Отрисовка детекций и треков
        vis_frame = self.tracker.draw_tracks(vis_frame, active_tracks)
        
        # 2. Отрисовка анализа поведения
        behavior_features = type('BehaviorFeatures', (), result.get('behavior_features', {}))()
        vis_frame = self.behavior_analyzer.draw_behavior_analysis(
            vis_frame, active_tracks, behavior_features
        )
        
        # 3. Информационная панель
        self._draw_info_panel(vis_frame, result)
        
        # 4. Индикатор риска
        self._draw_risk_indicator(vis_frame, result)
        
        return vis_frame
    
    def _draw_info_panel(self, frame: np.ndarray, result: dict):
        """Отрисовка информационной панели"""
        
        h, w = frame.shape[:2]
        panel_height = 150
        
        # Полупрозрачный фон для панели
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Текстовая информация
        info_lines = [
            f"Frame: {result['frame_id']} | Time: {result['processing_time']:.3f}s",
            f"Detections: {result['detections']} | Tracks: {result['tracks']}",
            f"Convoy: {result['behavior_features']['convoy_flag']} (size: {result['behavior_features']['convoy_size']})",
            f"Anomalies: Speed={result['behavior_features']['anomaly_speed']}, Dir={result['behavior_features']['anomaly_direction']}",
            f"Density: {result['behavior_features']['density']:.3f} | Avg Speed: {result['behavior_features']['avg_scene_speed']:.1f}",
            f"Thermal: {result['thermal_data']['available']} | Score: {result['thermal_data']['thermal_anomaly_score']:.2f}"
        ]
        
        y_offset = h - panel_height + 10
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
    
    def _draw_risk_indicator(self, frame: np.ndarray, result: dict):
        """Отрисовка индикатора риска"""
        
        risk_score = result['risk_assessment']['risk_score']
        risk_level = result['risk_assessment']['risk_level']
        
        # Цвет риска
        if risk_level == 'normal':
            color = (0, 255, 0)  # Зеленый
        elif risk_level == 'monitor':
            color = (0, 255, 255)  # Желтый
        else:  # alert
            color = (0, 0, 255)  # Красный
        
        # Рисуем индикатор
        h, w = frame.shape[:2]
        indicator_width = int(w * 0.8)
        indicator_height = 20
        x = (w - indicator_width) // 2
        y = 30
        
        # Фон
        cv2.rectangle(frame, (x, y), (x + indicator_width, y + indicator_height), 
                     (50, 50, 50), -1)
        
        # Заполнение риска
        fill_width = int(indicator_width * risk_score)
        cv2.rectangle(frame, (x, y), (x + fill_width, y + indicator_height), 
                     color, -1)
        
        # Текст
        risk_text = f"RISK: {risk_level.upper()} ({risk_score:.2f})"
        text_size = cv2.getTextSize(risk_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (w - text_size[0]) // 2
        
        cv2.putText(frame, risk_text, (text_x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Событие
        if result['event']['generated']:
            event_text = f"EVENT: {result['event']['event_type'].upper()} - {result['event']['severity'].upper()}"
            cv2.putText(frame, event_text, (10, y + indicator_height + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    def get_pipeline_statistics(self) -> dict:
        """Получение статистики пайплайна"""
        
        # Статистика модулей
        detector_stats = {
            'avg_processing_time': self.detector.get_average_processing_time()
        }
        
        tracker_stats = self.tracker.get_statistics()
        fusion_stats = self.fusion_engine.get_statistics()
        risk_stats = self.risk_engine.get_statistics()
        event_stats = self.event_generator.get_statistics()
        
        # Общая статистика
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        return {
            'pipeline': {
                'frames_processed': self.frame_count,
                'avg_processing_time': avg_processing_time,
                'fps': fps,
                'is_running': self.is_running
            },
            'detector': detector_stats,
            'tracker': tracker_stats,
            'fusion': fusion_stats,
            'risk': risk_stats,
            'events': event_stats
        }
    
    def run_camera_loop(self, camera_id: int = 0):
        """Основной цикл с камерой"""
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return
        
        print(f"Starting pipeline with camera {camera_id}")
        print("Press 'q' to quit, 's' for statistics, 'e' for events")
        
        self.is_running = True
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Обрабатываем кадр
            result = self.process_frame(frame)
            
            # Создаем визуализацию
            vis_frame = self.create_visualization(frame, result)
            
            # Показываем результат
            cv2.imshow('Distributed AI Monitoring System', vis_frame)
            
            # Обработка клавиатуры
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._print_statistics()
            elif key == ord('e'):
                self._print_recent_events()
        
        # Остановка
        self.is_running = False
        cap.release()
        cv2.destroyAllWindows()
        print("Pipeline stopped")
    
    def _print_statistics(self):
        """Печать статистики"""
        stats = self.get_pipeline_statistics()
        print("\n" + "="*50)
        print("PIPELINE STATISTICS")
        print("="*50)
        print(json.dumps(stats, indent=2, default=str))
        print("="*50)
    
    def _print_recent_events(self):
        """Печать последних событий"""
        recent_events = self.event_generator.get_recent_events(300)  # 5 минут
        print(f"\nRECENT EVENTS (last 5 minutes): {len(recent_events)}")
        for event in recent_events[-5:]:  # Последние 5 событий
            print(f"  {event.event_type.value} - {event.severity.value} - Score: {event.risk_score:.2f}")
    
    def export_all_data(self, base_filename: str = None):
        """Экспорт всех данных"""
        
        if base_filename is None:
            base_filename = f"pipeline_export_{int(time.time())}"
        
        # Экспорт данных из каждого модуля
        fusion_file = self.fusion_engine.export_fusion_data(f"{base_filename}_fusion.json")
        risk_file = self.risk_engine.export_risk_data(f"{base_filename}_risk.json")
        events_file = self.event_generator.export_events(f"{base_filename}_events.json")
        
        # Статистика
        stats = self.get_pipeline_statistics()
        stats_file = f"{base_filename}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"Data exported:")
        print(f"  Fusion: {fusion_file}")
        print(f"  Risk: {risk_file}")
        print(f"  Events: {events_file}")
        print(f"  Stats: {stats_file}")

def main():
    """Главная функция"""
    
    print("="*60)
    print("DISTRIBUTED AI SATELLITE MONITORING SYSTEM")
    print("="*60)
    print("Pipeline: Detection → Tracking → Behavior → Fusion → Risk → Events")
    print("Location: Moscow (55.7558, 37.6173)")
    print("="*60)
    
    # Создаем пайплайн
    pipeline = MonitoringPipeline(location=(55.7558, 37.6173))
    
    try:
        # Запускаем основной цикл
        pipeline.run_camera_loop(camera_id=0)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Экспорт данных при завершении
        print("\nExporting data...")
        pipeline.export_all_data()
        print("System shutdown complete")

if __name__ == "__main__":
    main()
