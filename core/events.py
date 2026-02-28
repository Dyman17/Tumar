"""
Event Generator - Генерация событий при RiskScore > 0.6
Создает объект события и сохраняет в БД
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import time
import uuid
import json
from datetime import datetime

from risk import RiskAssessment, RiskLevel
from tracker import Track
from fusion import FusedFeatures

class EventSeverity(Enum):
    """Уровень серьезности события"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EventType(Enum):
    """Типы событий"""
    CONVOY_DETECTED = "convoy_detected"
    SPEED_ANOMALY = "speed_anomaly"
    DIRECTION_ANOMALY = "direction_anomaly"
    HIGH_DENSITY = "high_density"
    THERMAL_ANOMALY = "thermal_anomaly"
    COMPOSITE_THREAT = "composite_threat"

@dataclass
class Event:
    """Класс события"""
    event_id: str
    timestamp: float
    event_type: EventType
    severity: EventSeverity
    risk_score: float
    location: Optional[tuple]
    detected_objects: List[Dict[str, Any]]
    reason: List[str]
    description: str
    metadata: Dict[str, Any]
    is_active: bool = True
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None

class EventGenerator:
    """Генератор событий"""
    
    def __init__(self):
        # Порог генерации событий
        self.event_threshold = 0.6
        
        # База данных событий (в реальной системе - PostgreSQL/MongoDB)
        self.events_db = []
        self.active_events = {}  # {event_id: Event}
        
        # Статистика
        self.total_events = 0
        self.events_by_type = {}
        self.events_by_severity = {}
        
        # Правила генерации событий
        self.event_rules = {
            'convoy_flag': {
                'event_type': EventType.CONVOY_DETECTED,
                'severity': EventSeverity.MEDIUM,
                'min_convoy_size': 3
            },
            'anomaly_speed': {
                'event_type': EventType.SPEED_ANOMALY,
                'severity': EventSeverity.HIGH
            },
            'anomaly_direction': {
                'event_type': EventType.DIRECTION_ANOMALY,
                'severity': EventSeverity.MEDIUM
            },
            'density': {
                'event_type': EventType.HIGH_DENSITY,
                'severity': EventSeverity.LOW,
                'threshold': 0.02
            },
            'thermal_activity': {
                'event_type': EventType.THERMAL_ANOMALY,
                'severity': EventSeverity.HIGH
            }
        }
    
    def should_generate_event(self, risk_assessment: RiskAssessment) -> bool:
        """Проверка, нужно ли генерировать событие"""
        return risk_assessment.risk_score >= self.event_threshold
    
    def generate_event(self, risk_assessment: RiskAssessment, 
                      tracks: List[Track], fused_features: FusedFeatures,
                      location: Optional[tuple] = None) -> Optional[Event]:
        """Генерация события на основе оценки риска"""
        
        if not self.should_generate_event(risk_assessment):
            return None
        
        # Определяем тип и серьезность события
        event_type, severity, reasons = self._determine_event_details(
            risk_assessment, fused_features
        )
        
        # Создаем объект события
        event = Event(
            event_id=str(uuid.uuid4()),
            timestamp=risk_assessment.timestamp,
            event_type=event_type,
            severity=severity,
            risk_score=risk_assessment.risk_score,
            location=location,
            detected_objects=self._extract_detected_objects(tracks),
            reason=reasons,
            description=self._generate_description(event_type, reasons, risk_assessment),
            metadata={
                'risk_assessment': asdict(risk_assessment),
                'fused_features': asdict(fused_features),
                'track_count': len(tracks),
                'generation_timestamp': time.time()
            }
        )
        
        # Сохраняем событие
        self._save_event(event)
        
        return event
    
    def _determine_event_details(self, risk_assessment: RiskAssessment, 
                               fused_features: FusedFeatures) -> tuple[EventType, EventSeverity, List[str]]:
        """Определение типа, серьезности и причин события"""
        
        # Анализируем основные факторы риска
        primary_factors = risk_assessment.primary_factors
        reasons = []
        
        # Определяем тип события на основе основного фактора
        if 'convoy_flag' in primary_factors:
            event_type = EventType.CONVOY_DETECTED
            reasons.append(f"Vehicle convoy detected (size: {fused_features.convoy_size})")
        elif 'anomaly_speed' in primary_factors:
            event_type = EventType.SPEED_ANOMALY
            reasons.append("Anomalous vehicle speed detected")
        elif 'anomaly_direction' in primary_factors:
            event_type = EventType.DIRECTION_ANOMALY
            reasons.append("Unusual direction changes detected")
        elif 'thermal_activity' in primary_factors:
            event_type = EventType.THERMAL_ANOMALY
            reasons.append("Thermal activity detected in vehicle zones")
        else:
            # Композитная угроза
            event_type = EventType.COMPOSITE_THREAT
            reasons.extend([f"Multiple risk factors: {', '.join(primary_factors)}"])
        
        # Добавляем дополнительные причины
        if 'density' in primary_factors:
            reasons.append(f"High object density: {fused_features.density:.3f}")
        
        # Определяем серьезность на основе риска
        if risk_assessment.risk_score >= 0.8:
            severity = EventSeverity.CRITICAL
        elif risk_assessment.risk_score >= 0.7:
            severity = EventSeverity.HIGH
        elif risk_assessment.risk_score >= 0.6:
            severity = EventSeverity.MEDIUM
        else:
            severity = EventSeverity.LOW
        
        # Повышаем серьезность для определенных типов
        if event_type in [EventType.CONVOY_DETECTED, EventType.THERMAL_ANOMALY]:
            if severity == EventSeverity.MEDIUM:
                severity = EventSeverity.HIGH
            elif severity == EventSeverity.LOW:
                severity = EventSeverity.MEDIUM
        
        return event_type, severity, reasons
    
    def _extract_detected_objects(self, tracks: List[Track]) -> List[Dict[str, Any]]:
        """Извлечение информации о детектированных объектах"""
        
        detected_objects = []
        
        for track in tracks:
            obj_info = {
                'id': track.id,
                'class': track.class_name,
                'center': track.center,
                'velocity': track.velocity,
                'acceleration': track.acceleration,
                'bbox': track.bbox,
                'confidence': track.confidence,
                'trajectory_length': len(track.trajectory)
            }
            detected_objects.append(obj_info)
        
        return detected_objects
    
    def _generate_description(self, event_type: EventType, reasons: List[str], 
                           risk_assessment: RiskAssessment) -> str:
        """Генерация описания события"""
        
        base_descriptions = {
            EventType.CONVOY_DETECTED: "Vehicle convoy formation detected",
            EventType.SPEED_ANOMALY: "Anomalous speed patterns detected",
            EventType.DIRECTION_ANOMALY: "Unusual movement direction changes",
            EventType.HIGH_DENSITY: "High object density detected",
            EventType.THERMAL_ANOMALY: "Thermal anomalies in monitored area",
            EventType.COMPOSITE_THREAT: "Multiple threat factors detected"
        }
        
        base_desc = base_descriptions.get(event_type, "Security event detected")
        
        # Добавляем детали
        details = f". Risk score: {risk_assessment.risk_score:.2f}. "
        details += f"Primary factors: {', '.join(risk_assessment.primary_factors)}. "
        
        if reasons:
            details += f"Details: {'; '.join(reasons)}."
        
        return base_desc + details
    
    def _save_event(self, event: Event):
        """Сохранение события в БД"""
        
        # В реальной системе здесь будет запись в PostgreSQL/MongoDB
        self.events_db.append(event)
        self.active_events[event.event_id] = event
        
        # Обновляем статистику
        self.total_events += 1
        
        # Статистика по типам
        event_type_str = event.event_type.value
        self.events_by_type[event_type_str] = self.events_by_type.get(event_type_str, 0) + 1
        
        # Статистика по серьезности
        severity_str = event.severity.value
        self.events_by_severity[severity_str] = self.events_by_severity.get(severity_str, 0) + 1
        
        print(f"[EVENT] {event.event_type.value.upper()} - {event.severity.value.upper()} - Score: {event.risk_score:.2f}")
        print(f"  ID: {event.event_id}")
        print(f"  Description: {event.description}")
        print(f"  Reasons: {event.reason}")
    
    def acknowledge_event(self, event_id: str, acknowledged_by: str = "system") -> bool:
        """Подтверждение события"""
        
        if event_id in self.active_events:
            event = self.active_events[event_id]
            event.acknowledged = True
            event.acknowledged_by = acknowledged_by
            event.acknowledged_at = time.time()
            
            print(f"[ACK] Event {event_id} acknowledged by {acknowledged_by}")
            return True
        
        return False
    
    def get_active_events(self) -> List[Event]:
        """Получение активных событий"""
        return list(self.active_events.values())
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Получение событий по типу"""
        return [event for event in self.events_db if event.event_type == event_type]
    
    def get_events_by_severity(self, severity: EventSeverity) -> List[Event]:
        """Получение событий по серьезности"""
        return [event for event in self.events_db if event.severity == severity]
    
    def get_recent_events(self, time_window: float = 3600.0) -> List[Event]:
        """Получение событий за временной интервал"""
        
        current_time = time.time()
        return [
            event for event in self.events_db
            if current_time - event.timestamp <= time_window
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики событий"""
        
        # Активные события
        active_count = len(self.active_events)
        
        # События за последний час
        recent_events = self.get_recent_events(3600.0)
        recent_count = len(recent_events)
        
        # Распределение по серьезности
        severity_distribution = {}
        for severity in EventSeverity:
            count = len([e for e in self.events_db if e.severity == severity])
            severity_distribution[severity.value] = count
        
        # Распределение по типам
        type_distribution = {}
        for event_type in EventType:
            count = len([e for e in self.events_db if e.event_type == event_type])
            type_distribution[event_type.value] = count
        
        # Средний риск для событий
        if self.events_db:
            avg_risk = sum(e.risk_score for e in self.events_db) / len(self.events_db)
            max_risk = max(e.risk_score for e in self.events_db)
        else:
            avg_risk = 0.0
            max_risk = 0.0
        
        return {
            'total_events': self.total_events,
            'active_events': active_count,
            'recent_events': recent_count,
            'severity_distribution': severity_distribution,
            'type_distribution': type_distribution,
            'average_risk_score': avg_risk,
            'max_risk_score': max_risk,
            'acknowledged_events': len([e for e in self.events_db if e.acknowledged]),
            'acknowledgment_rate': len([e for e in self.events_db if e.acknowledged]) / len(self.events_db) if self.events_db else 0
        }
    
    def export_events(self, filename: str = None) -> str:
        """Экспорт событий"""
        
        if filename is None:
            filename = f"events_{int(time.time())}.json"
        
        export_data = {
            'metadata': {
                'export_timestamp': time.time(),
                'total_events': len(self.events_db),
                'event_threshold': self.event_threshold
            },
            'events': [
                {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'event_type': event.event_type.value,
                    'severity': event.severity.value,
                    'risk_score': event.risk_score,
                    'location': event.location,
                    'detected_objects': event.detected_objects,
                    'reason': event.reason,
                    'description': event.description,
                    'is_active': event.is_active,
                    'acknowledged': event.acknowledged,
                    'acknowledged_by': event.acknowledged_by,
                    'acknowledged_at': event.acknowledged_at,
                    'metadata': event.metadata
                }
                for event in self.events_db
            ],
            'statistics': self.get_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename

if __name__ == "__main__":
    # Тестирование Event Generator
    from risk import RiskEngine, RiskLevel
    from fusion import FusedFeatures
    from tracker import Track
    
    event_generator = EventGenerator()
    
    print("Testing Event Generator...")
    
    # Создаем тестовые данные
    test_tracks = [
        Track(
            id=1, class_name='truck', center=(100, 100), velocity=(15, 5),
            acceleration=(0, 0), bbox=(80, 80, 120, 120), confidence=0.9,
            last_seen=time.time(), trajectory=[], disappeared_count=0,
            kalman_filter=None
        ),
        Track(
            id=2, class_name='truck', center=(150, 110), velocity=(14, 6),
            acceleration=(0, 0), bbox=(130, 90, 170, 130), confidence=0.8,
            last_seen=time.time(), trajectory=[], disappeared_count=0,
            kalman_filter=None
        )
    ]
    
    # Тестовый сценарий высокого риска
    test_fused_features = FusedFeatures(
        convoy_flag=True, convoy_size=2, anomaly_speed=True,
        anomaly_direction=False, density=0.025, thermal_activity=True,
        thermal_data_available=True, timestamp=time.time()
    )
    
    try:
        # Создаем оценку риска
        risk_engine = RiskEngine()
        risk_assessment = risk_engine.calculate_risk(test_fused_features)
        
        print(f"Risk Assessment: {risk_assessment.risk_score:.3f} ({risk_assessment.risk_level.value})")
        
        # Генерируем событие
        event = event_generator.generate_event(
            risk_assessment, test_tracks, test_fused_features, 
            location=(55.7558, 37.6173)
        )
        
        if event:
            print(f"\nEvent Generated:")
            print(f"  ID: {event.event_id}")
            print(f"  Type: {event.event_type.value}")
            print(f"  Severity: {event.severity.value}")
            print(f"  Risk Score: {event.risk_score:.3f}")
            print(f"  Description: {event.description}")
            print(f"  Reasons: {event.reason}")
            print(f"  Detected Objects: {len(event.detected_objects)}")
        else:
            print("No event generated (risk below threshold)")
        
        # Тест с низким риском
        low_risk_features = FusedFeatures(
            convoy_flag=False, convoy_size=0, anomaly_speed=False,
            anomaly_direction=False, density=0.005, thermal_activity=False,
            thermal_data_available=True, timestamp=time.time()
        )
        
        low_risk_assessment = risk_engine.calculate_risk(low_risk_features)
        print(f"\nLow Risk Assessment: {low_risk_assessment.risk_score:.3f} ({low_risk_assessment.risk_level.value})")
        
        low_event = event_generator.generate_event(
            low_risk_assessment, [], low_risk_features
        )
        
        if not low_event:
            print("No event generated (expected for low risk)")
        
        # Статистика
        stats = event_generator.get_statistics()
        print(f"\nEvent Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Активные события
        active_events = event_generator.get_active_events()
        print(f"\nActive Events: {len(active_events)}")
        for event in active_events:
            print(f"  - {event.event_type.value}: {event.severity.value}")
        
        # Экспорт событий
        export_file = event_generator.export_events()
        print(f"\nEvents exported to: {export_file}")
        
        print("\nEvent Generator test completed successfully!")
        
    except Exception as e:
        print(f"Error during event generator test: {e}")
        import traceback
        traceback.print_exc()
