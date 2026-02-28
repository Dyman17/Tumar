import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import time
import json
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from enum import Enum
import math
from scipy import stats

from data_fusion import FusedData, SensorData, EnvironmentalData, SatelliteData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Уровни риска"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    """Типы алертов"""
    SECURITY = "security"
    SAFETY = "safety"
    ENVIRONMENTAL = "environmental"
    TECHNICAL = "technical"
    HEALTH = "health"

@dataclass
class RiskFactor:
    """Фактор риска"""
    name: str
    weight: float
    current_value: float
    threshold: float
    severity: str
    description: str
    timestamp: float

@dataclass
class RiskAssessment:
    """Оценка риска"""
    total_score: float
    risk_level: RiskLevel
    confidence: float
    primary_factors: List[str]
    secondary_factors: List[str]
    recommendations: List[str]
    timestamp: float
    metadata: Dict

@dataclass
class Alert:
    """Алерт"""
    alert_id: str
    alert_type: AlertType
    severity: str
    title: str
    description: str
    risk_score: float
    location: Optional[Tuple[float, float]]
    timestamp: float
    is_active: bool
    metadata: Dict

class RiskEngine:
    """Основной движок оценки риска"""
    
    def __init__(self, location: Tuple[float, float] = None):
        self.location = location
        
        # Веса факторов риска
        self.risk_weights = {
            'object_density': 0.20,
            'anomaly_detection': 0.25,
            'thermal_activity': 0.15,
            'fire_detection': 0.20,
            'environmental': 0.10,
            'historical_patterns': 0.10
        }
        
        # Пороги для разных уровней
        self.risk_thresholds = {
            RiskLevel.VERY_LOW: 0.0,
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.4,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 0.85
        }
        
        # История оценок
        self.assessment_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=500)
        
        # База знаний для рекомендаций
        self.recommendations_db = self._initialize_recommendations()
        
        # Статистические модели
        self.baseline_models = {}
        self.anomaly_thresholds = {}
        
    def _initialize_recommendations(self) -> Dict:
        """Инициализация базы рекомендаций"""
        return {
            'high_object_density': [
                "Increase monitoring frequency",
                "Deploy additional sensors",
                "Consider crowd control measures"
            ],
            'thermal_anomaly': [
                "Verify heat source",
                "Check for fire hazards",
                "Inspect electrical equipment"
            ],
            'anomaly_detected': [
                "Review recent activity patterns",
                "Increase surveillance",
                "Check system integrity"
            ],
            'adverse_weather': [
                "Secure outdoor equipment",
                "Postpone non-critical activities",
                "Monitor weather updates"
            ],
            'critical_risk': [
                "Immediate notification of authorities",
                "Emergency protocol activation",
                "Area evacuation consideration"
            ]
        }
    
    def calculate_object_density_risk(self, fused_data: FusedData) -> RiskFactor:
        """Расчет риска плотности объектов"""
        
        detection_count = len(fused_data.local_detections)
        
        # Базовые пороги
        density_thresholds = {
            'LOW': 5,
            'MEDIUM': 10,
            'HIGH': 20,
            'CRITICAL': 30
        }
        
        # Нормализуем значение
        max_expected = 50  # Максимальное ожидаемое количество объектов
        normalized_density = min(detection_count / max_expected, 1.0)
        
        # Определяем серьезность
        if detection_count < density_thresholds['LOW']:
            severity = 'LOW'
        elif detection_count < density_thresholds['MEDIUM']:
            severity = 'MEDIUM'
        elif detection_count < density_thresholds['HIGH']:
            severity = 'HIGH'
        else:
            severity = 'CRITICAL'
        
        return RiskFactor(
            name="object_density",
            weight=self.risk_weights['object_density'],
            current_value=normalized_density,
            threshold=0.5,
            severity=severity,
            description=f"Detected {detection_count} objects in monitoring area",
            timestamp=fused_data.timestamp
        )
    
    def calculate_anomaly_risk(self, fused_data: FusedData) -> RiskFactor:
        """Расчет риска аномалий"""
        
        anomaly_summary = fused_data.fusion_metadata.get('anomaly_summary', {})
        
        # Базовый score на основе наличия аномалий
        base_score = 0.0
        severity = 'LOW'
        
        if anomaly_summary.get('has_anomaly', False):
            anomaly_count = anomaly_summary.get('anomaly_count', 0)
            max_severity = anomaly_summary.get('max_severity', 'low')
            
            # Конвертируем серьезность в числовой score
            severity_scores = {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8,
                'critical': 1.0
            }
            
            base_score = severity_scores.get(max_severity.lower(), 0.3)
            severity = max_severity.upper()
            
            # Учитываем количество аномалий
            if anomaly_count > 1:
                base_score = min(base_score * (1 + anomaly_count * 0.1), 1.0)
        
        return RiskFactor(
            name="anomaly_detection",
            weight=self.risk_weights['anomaly_detection'],
            current_value=base_score,
            threshold=0.3,
            severity=severity,
            description=f"Anomaly detection: {anomaly_summary.get('anomaly_types', [])}",
            timestamp=fused_data.timestamp
        )
    
    def calculate_thermal_risk(self, fused_data: FusedData) -> Tuple[RiskFactor, RiskFactor]:
        """Расчет тепловых рисков"""
        
        thermal_factor = RiskFactor(
            name="thermal_activity",
            weight=self.risk_weights['thermal_activity'],
            current_value=0.0,
            threshold=0.4,
            severity='LOW',
            description="No thermal data available",
            timestamp=fused_data.timestamp
        )
        
        fire_factor = RiskFactor(
            name="fire_detection",
            weight=self.risk_weights['fire_detection'],
            current_value=0.0,
            threshold=0.3,
            severity='LOW',
            description="No fire detection data available",
            timestamp=fused_data.timestamp
        )
        
        if fused_data.satellite_data:
            # Тепловые аномалии
            thermal_score = fused_data.satellite_data.thermal_anomaly_score
            thermal_factor.current_value = thermal_score
            
            if thermal_score > 0.8:
                thermal_factor.severity = 'CRITICAL'
                thermal_factor.description = f"Critical thermal anomaly detected: {thermal_score:.2f}"
            elif thermal_score > 0.6:
                thermal_factor.severity = 'HIGH'
                thermal_factor.description = f"High thermal activity: {thermal_score:.2f}"
            elif thermal_score > 0.4:
                thermal_factor.severity = 'MEDIUM'
                thermal_factor.description = f"Moderate thermal activity: {thermal_score:.2f}"
            else:
                thermal_factor.severity = 'LOW'
                thermal_factor.description = f"Normal thermal levels: {thermal_score:.2f}"
            
            # Детекция огня
            fire_score = fused_data.satellite_data.fire_detection_score
            fire_factor.current_value = fire_score
            
            if fire_score > 0.7:
                fire_factor.severity = 'CRITICAL'
                fire_factor.description = f"Fire detected: {fire_score:.2f}"
            elif fire_score > 0.4:
                fire_factor.severity = 'HIGH'
                fire_factor.description = f"Possible fire activity: {fire_score:.2f}"
            else:
                fire_factor.severity = 'LOW'
                fire_factor.description = f"No fire detected: {fire_score:.2f}"
        
        return thermal_factor, fire_factor
    
    def calculate_environmental_risk(self, fused_data: FusedData) -> RiskFactor:
        """Расчет environmental риска"""
        
        env_factor = RiskFactor(
            name="environmental",
            weight=self.risk_weights['environmental'],
            current_value=0.0,
            threshold=0.3,
            severity='LOW',
            description="Normal environmental conditions",
            timestamp=fused_data.timestamp
        )
        
        if fused_data.environmental_data:
            env = fused_data.environmental_data
            risk_score = 0.0
            risk_factors = []
            
            # Экстремальные температуры
            if env.temperature > 35:
                risk_score += 0.3
                risk_factors.append("High temperature")
            elif env.temperature < -10:
                risk_score += 0.3
                risk_factors.append("Low temperature")
            
            # Низкая видимость
            if env.visibility < 1.0:
                risk_score += 0.2
                risk_factors.append("Low visibility")
            
            # Сильный ветер
            if env.wind_speed > 15:
                risk_score += 0.2
                risk_factors.append("Strong wind")
            
            # Неблагоприятные погодные условия
            adverse_conditions = ['storm', 'fog', 'heavy rain', 'snow', 'hail']
            if any(condition in env.weather_condition.lower() 
                   for condition in adverse_conditions):
                risk_score += 0.3
                risk_factors.append("Adverse weather")
            
            env_factor.current_value = min(risk_score, 1.0)
            
            if risk_score > 0.6:
                env_factor.severity = 'HIGH'
            elif risk_score > 0.3:
                env_factor.severity = 'MEDIUM'
            else:
                env_factor.severity = 'LOW'
            
            env_factor.description = f"Weather risks: {', '.join(risk_factors) if risk_factors else 'None'}"
        
        return env_factor
    
    def calculate_historical_pattern_risk(self, fused_data: FusedData) -> RiskFactor:
        """Расчет риска на основе исторических паттернов"""
        
        historical_factor = RiskFactor(
            name="historical_patterns",
            weight=self.risk_weights['historical_patterns'],
            current_value=0.0,
            threshold=0.2,
            severity='LOW',
            description="No historical pattern analysis",
            timestamp=fused_data.timestamp
        )
        
        if len(self.assessment_history) < 10:
            return historical_factor
        
        # Анализ последних оценок
        recent_assessments = list(self.assessment_history)[-20:]
        recent_scores = [a.total_score for a in recent_assessments]
        
        # Проверяем тренд
        if len(recent_scores) >= 5:
            recent_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            # Если тренд восходящий
            if recent_trend > 0.01:
                historical_factor.current_value = min(recent_trend * 10, 1.0)
                historical_factor.severity = 'MEDIUM' if recent_trend > 0.02 else 'LOW'
                historical_factor.description = f"Upward risk trend detected: {recent_trend:.3f}"
            else:
                historical_factor.current_value = 0.0
                historical_factor.severity = 'LOW'
                historical_factor.description = "Stable or decreasing risk trend"
        
        return historical_factor
    
    def assess_comprehensive_risk(self, fused_data: FusedData) -> RiskAssessment:
        """Комплексная оценка риска"""
        
        # Расчет всех факторов
        risk_factors = []
        
        # Плотность объектов
        density_factor = self.calculate_object_density_risk(fused_data)
        risk_factors.append(density_factor)
        
        # Аномалии
        anomaly_factor = self.calculate_anomaly_risk(fused_data)
        risk_factors.append(anomaly_factor)
        
        # Тепловые риски
        thermal_factor, fire_factor = self.calculate_thermal_risk(fused_data)
        risk_factors.extend([thermal_factor, fire_factor])
        
        # Environmental риски
        env_factor = self.calculate_environmental_risk(fused_data)
        risk_factors.append(env_factor)
        
        # Исторические паттерны
        historical_factor = self.calculate_historical_pattern_risk(fused_data)
        risk_factors.append(historical_factor)
        
        # Расчет общего риска
        total_risk = sum(factor.current_value * factor.weight for factor in risk_factors)
        
        # Определение уровня риска
        risk_level = self._determine_risk_level(total_risk)
        
        # Расчет уверенности
        confidence = self._calculate_assessment_confidence(fused_data, risk_factors)
        
        # Идентификация ключевых факторов
        primary_factors = [f.name for f in risk_factors if f.current_value > 0.5]
        secondary_factors = [f.name for f in risk_factors if 0.2 < f.current_value <= 0.5]
        
        # Генерация рекомендаций
        recommendations = self._generate_recommendations(risk_factors, risk_level)
        
        assessment = RiskAssessment(
            total_score=total_risk,
            risk_level=risk_level,
            confidence=confidence,
            primary_factors=primary_factors,
            secondary_factors=secondary_factors,
            recommendations=recommendations,
            timestamp=fused_data.timestamp,
            metadata={
                'risk_factors': [asdict(f) for f in risk_factors],
                'location': fused_data.location,
                'detection_count': len(fused_data.local_detections),
                'anomaly_count': len(fused_data.anomaly_results)
            }
        )
        
        # Сохраняем в историю
        self.assessment_history.append(assessment)
        
        # Генерируем алерты если нужно
        self._generate_alerts(assessment)
        
        return assessment
    
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Определение уровня риска"""
        for level, threshold in sorted(self.risk_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return level
        return RiskLevel.VERY_LOW
    
    def _calculate_assessment_confidence(self, fused_data: FusedData, 
                                       risk_factors: List[RiskFactor]) -> float:
        """Расчет уверенности в оценке"""
        confidence = 0.3  # Базовая уверенность
        
        # Наличие локальных детекций
        if fused_data.local_detections:
            confidence += 0.2
        
        # Наличие спутниковых данных
        if fused_data.satellite_data:
            confidence += 0.2
        
        # Наличие environmental данных
        if fused_data.environmental_data:
            confidence += 0.1
        
        # Количество факторов с данными
        factors_with_data = sum(1 for f in risk_factors if f.current_value > 0)
        confidence += min(factors_with_data * 0.05, 0.2)
        
        return min(confidence, 1.0)
    
    def _generate_recommendations(self, risk_factors: List[RiskFactor], 
                                risk_level: RiskLevel) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        
        # Рекомендации на основе факторов
        for factor in risk_factors:
            if factor.current_value > 0.5:
                if factor.name == 'object_density':
                    recommendations.extend(self.recommendations_db['high_object_density'])
                elif factor.name in ['thermal_activity', 'fire_detection']:
                    recommendations.extend(self.recommendations_db['thermal_anomaly'])
                elif factor.name == 'anomaly_detection':
                    recommendations.extend(self.recommendations_db['anomaly_detected'])
                elif factor.name == 'environmental':
                    recommendations.extend(self.recommendations_db['adverse_weather'])
        
        # Рекомендации на основе уровня риска
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend(self.recommendations_db['critical_risk'])
        
        # Удаляем дубликаты
        recommendations = list(set(recommendations))
        
        # Ограничиваем количество
        return recommendations[:5]
    
    def _generate_alerts(self, assessment: RiskAssessment):
        """Генерация алертов"""
        
        # Алерт для критического риска
        if assessment.risk_level == RiskLevel.CRITICAL:
            alert = Alert(
                alert_id=f"critical_{int(assessment.timestamp)}",
                alert_type=AlertType.SECURITY,
                severity="CRITICAL",
                title="Critical Risk Level Detected",
                description=f"Risk score: {assessment.total_score:.2f}. Immediate attention required.",
                risk_score=assessment.total_score,
                location=self.location,
                timestamp=assessment.timestamp,
                is_active=True,
                metadata={
                    'primary_factors': assessment.primary_factors,
                    'recommendations': assessment.recommendations
                }
            )
            self.alert_history.append(alert)
        
        # Алерт для высокого риска
        elif assessment.risk_level == RiskLevel.HIGH:
            alert = Alert(
                alert_id=f"high_{int(assessment.timestamp)}",
                alert_type=AlertType.SECURITY,
                severity="HIGH",
                title="High Risk Level Detected",
                description=f"Risk score: {assessment.total_score:.2f}. Increased monitoring recommended.",
                risk_score=assessment.total_score,
                location=self.location,
                timestamp=assessment.timestamp,
                is_active=True,
                metadata={
                    'primary_factors': assessment.primary_factors
                }
            )
            self.alert_history.append(alert)
    
    def get_risk_trends(self, hours: int = 24) -> Dict:
        """Получение трендов риска"""
        
        cutoff_time = time.time() - (hours * 3600)
        recent_assessments = [a for a in self.assessment_history 
                            if a.timestamp > cutoff_time]
        
        if len(recent_assessments) < 2:
            return {'trend': 'insufficient_data', 'slope': 0.0}
        
        # Расчет тренда
        times = [(a.timestamp - recent_assessments[0].timestamp) / 3600 
                for a in recent_assessments]  # часы от начала
        scores = [a.total_score for a in recent_assessments]
        
        # Линейная регрессия
        if len(times) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(times, scores)
            
            # Определяем тренд
            if abs(slope) < 0.01:
                trend = 'stable'
            elif slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            
            return {
                'trend': trend,
                'slope': slope,
                'correlation': r_value,
                'significance': p_value,
                'data_points': len(recent_assessments)
            }
        
        return {'trend': 'stable', 'slope': 0.0}
    
    def get_active_alerts(self) -> List[Alert]:
        """Получение активных алертов"""
        return [alert for alert in self.alert_history if alert.is_active]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Подтверждение алерта"""
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.is_active = False
                return True
        return False
    
    def get_risk_statistics(self) -> Dict:
        """Получение статистики риска"""
        
        if not self.assessment_history:
            return {}
        
        all_scores = [a.total_score for a in self.assessment_history]
        recent_scores = [a.total_score for a in list(self.assessment_history)[-100:]]
        
        # Распределение уровней риска
        level_counts = defaultdict(int)
        for assessment in self.assessment_history:
            level_counts[assessment.risk_level.value] += 1
        
        return {
            'total_assessments': len(self.assessment_history),
            'current_risk': all_scores[-1] if all_scores else 0.0,
            'average_risk': np.mean(all_scores),
            'recent_average': np.mean(recent_scores) if recent_scores else 0.0,
            'max_risk': np.max(all_scores),
            'min_risk': np.min(all_scores),
            'risk_level_distribution': dict(level_counts),
            'active_alerts': len(self.get_active_alerts()),
            'total_alerts': len(self.alert_history)
        }
    
    def export_risk_data(self, filename: str = None) -> str:
        """Экспорт данных о рисках"""
        if filename is None:
            filename = f"risk_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'location': self.location,
                'total_assessments': len(self.assessment_history)
            },
            'risk_assessments': [asdict(a) for a in self.assessment_history],
            'alerts': [asdict(a) for a in self.alert_history],
            'statistics': self.get_risk_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filename

if __name__ == "__main__":
    # Тестирование Risk Engine
    print("Testing Risk Engine...")
    
    risk_engine = RiskEngine(location=(55.7558, 37.6173))
    
    # Создаем тестовые fused data
    from data_fusion import FusedData, Detection, AnomalyResult, SensorData, EnvironmentalData, SatelliteData
    
    test_fused_data = FusedData(
        timestamp=time.time(),
        local_detections=[
            Detection(class_name='person', confidence=0.8, bbox=(0, 0, 100, 100), center=(50, 50))
        ] * 8,  # 8 объектов
        anomaly_results=[
            AnomalyResult(
                is_anomaly=True,
                anomaly_score=0.7,
                anomaly_type="statistical",
                confidence=0.8,
                description="Unusual activity pattern",
                timestamp=time.time(),
                severity="medium"
            )
        ],
        sensor_data=SensorData(
            timestamp=time.time(),
            temperature=35.5,
            humidity=40.0,
            light_level=800
        ),
        environmental_data=EnvironmentalData(
            timestamp=time.time(),
            weather_condition="clear",
            temperature=36.0,
            humidity=35.0,
            wind_speed=5.0,
            wind_direction=180.0,
            visibility=10.0
        ),
        satellite_data=SatelliteData(
            timestamp=time.time(),
            thermal_anomaly_score=0.6,
            fire_detection_score=0.1,
            surface_temperature=40.0,
            ndvi_index=0.7,
            cloud_cover=0.2,
            data_source="NASA_GIBS"
        ),
        location=(55.7558, 37.6173),
        fusion_metadata={}
    )
    
    try:
        # Оценка риска
        assessment = risk_engine.assess_comprehensive_risk(test_fused_data)
        
        print(f"Risk Assessment Completed:")
        print(f"Total Risk Score: {assessment.total_score:.3f}")
        print(f"Risk Level: {assessment.risk_level.value}")
        print(f"Confidence: {assessment.confidence:.3f}")
        print(f"Primary Factors: {assessment.primary_factors}")
        print(f"Recommendations: {assessment.recommendations}")
        
        # Статистика
        stats = risk_engine.get_risk_statistics()
        print(f"\nRisk Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Тренды
        trends = risk_engine.get_risk_trends()
        print(f"\nRisk Trends: {trends}")
        
        # Активные алерты
        active_alerts = risk_engine.get_active_alerts()
        print(f"\nActive Alerts: {len(active_alerts)}")
        for alert in active_alerts:
            print(f"  - {alert.title}: {alert.severity}")
        
        # Экспорт данных
        export_file = risk_engine.export_risk_data()
        print(f"\nData exported to: {export_file}")
        
        print("\nRisk Engine test completed successfully!")
        
    except Exception as e:
        print(f"Error during risk engine testing: {e}")
        logger.exception("Risk engine test failed")
