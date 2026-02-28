"""
Risk Engine - Четкая формула оценки риска
RiskScore = 0.35*convoy_flag + 0.2*anomaly_speed + 0.15*anomaly_direction + 0.15*density + 0.15*thermal_activity
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import json

from fusion import FusedFeatures

class RiskLevel(Enum):
    """Уровни риска"""
    NORMAL = "normal"
    MONITOR = "monitor"
    ALERT = "alert"

@dataclass
class RiskAssessment:
    """Оценка риска"""
    risk_score: float
    risk_level: RiskLevel
    component_scores: Dict[str, float]
    primary_factors: List[str]
    timestamp: float

class RiskEngine:
    """Движок оценки риска с четкой формулой"""
    
    def __init__(self):
        # Четкая формула с весами
        self.weights = {
            'convoy_flag': 0.35,
            'anomaly_speed': 0.20,
            'anomaly_direction': 0.15,
            'density': 0.15,
            'thermal_activity': 0.15
        }
        
        # Пороги для уровней риска
        self.thresholds = {
            RiskLevel.NORMAL: 0.3,
            RiskLevel.MONITOR: 0.6,
            RiskLevel.ALERT: 1.0
        }
        
        # Пороги для компонентов
        self.component_thresholds = {
            'convoy_flag': 0.5,  # True/False
            'anomaly_speed': 0.5,  # True/False
            'anomaly_direction': 0.5,  # True/False
            'density': 0.01,  # объектов на пиксель
            'thermal_activity': 0.5  # True/False
        }
        
        # История оценок
        self.risk_history = []
        self.max_history_length = 1000
        
        # Статистика
        self.alert_count = 0
        self.total_assessments = 0
        
    def calculate_risk(self, fused_features: FusedFeatures) -> RiskAssessment:
        """Расчет риска по четкой формуле"""
        
        # 1. Нормализация компонентов
        normalized_components = self._normalize_components(fused_features)
        
        # 2. Расчет взвешенной суммы по формуле
        risk_score = sum(
            normalized_components[component] * weight
            for component, weight in self.weights.items()
        )
        
        # Ограничиваем диапазон [0, 1]
        risk_score = max(0.0, min(1.0, risk_score))
        
        # 3. Определение уровня риска
        risk_level = self._classify_risk_level(risk_score)
        
        # 4. Определение ключевых факторов
        primary_factors = self._identify_primary_factors(normalized_components)
        
        # 5. Создаем оценку
        assessment = RiskAssessment(
            risk_score=risk_score,
            risk_level=risk_level,
            component_scores=normalized_components,
            primary_factors=primary_factors,
            timestamp=fused_features.timestamp
        )
        
        # 6. Сохраняем в историю
        self.risk_history.append(assessment)
        if len(self.risk_history) > self.max_history_length:
            self.risk_history.pop(0)
        
        # 7. Обновляем статистику
        self.total_assessments += 1
        if risk_level == RiskLevel.ALERT:
            self.alert_count += 1
        
        return assessment
    
    def _normalize_components(self, fused_features: FusedFeatures) -> Dict[str, float]:
        """Нормализация компонентов к диапазону [0, 1]"""
        
        normalized = {}
        
        # 1. Конвой (бинарный)
        normalized['convoy_flag'] = 1.0 if fused_features.convoy_flag else 0.0
        
        # 2. Аномальная скорость (бинарный)
        normalized['anomaly_speed'] = 1.0 if fused_features.anomaly_speed else 0.0
        
        # 3. Аномальное направление (бинарный)
        normalized['anomaly_direction'] = 1.0 if fused_features.anomaly_direction else 0.0
        
        # 4. Плотность (непрерывная)
        # Нормализуем плотность относительно порога
        density_threshold = self.component_thresholds['density']
        normalized['density'] = min(fused_features.density / density_threshold, 1.0)
        
        # 5. Тепловая активность (бинарный)
        normalized['thermal_activity'] = 1.0 if fused_features.thermal_activity else 0.0
        
        return normalized
    
    def _classify_risk_level(self, risk_score: float) -> RiskLevel:
        """Классификация уровня риска по порогам"""
        
        if risk_score < self.thresholds[RiskLevel.NORMAL]:
            return RiskLevel.NORMAL
        elif risk_score < self.thresholds[RiskLevel.MONITOR]:
            return RiskLevel.MONITOR
        else:
            return RiskLevel.ALERT
    
    def _identify_primary_factors(self, component_scores: Dict[str, float]) -> List[str]:
        """Идентификация ключевых факторов риска"""
        
        # Сортируем компоненты по вкладу в риск (score * weight)
        factor_contributions = [
            (component, score * self.weights[component])
            for component, score in component_scores.items()
        ]
        
        # Сортируем по убыванию вклада
        factor_contributions.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем факторы с вкладом > 0.1
        primary_factors = [
            component for component, contribution in factor_contributions
            if contribution > 0.1
        ]
        
        return primary_factors
    
    def get_risk_trends(self, time_window: float = 300.0) -> Dict:
        """Анализ трендов риска"""
        
        current_time = time.time()
        recent_assessments = [
            assessment for assessment in self.risk_history
            if current_time - assessment.timestamp <= time_window
        ]
        
        if len(recent_assessments) < 2:
            return {'trend': 'insufficient_data'}
        
        # Анализируем тренд общего риска
        risk_scores = [assessment.risk_score for assessment in recent_assessments]
        
        # Сравниваем средние значения первой и второй половины
        mid_point = len(risk_scores) // 2
        first_half_avg = sum(risk_scores[:mid_point]) / mid_point if mid_point > 0 else 0
        second_half_avg = sum(risk_scores[mid_point:]) / (len(risk_scores) - mid_point) if len(risk_scores) - mid_point > 0 else 0
        
        # Определяем тренд
        if second_half_avg > first_half_avg + 0.1:
            trend = 'increasing'
        elif second_half_avg < first_half_avg - 0.1:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Анализ трендов компонентов
        component_trends = {}
        for component in self.weights.keys():
            component_scores = [
                assessment.component_scores.get(component, 0.0) 
                for assessment in recent_assessments
            ]
            
            if len(component_scores) >= 2:
                first_half_component = sum(component_scores[:mid_point]) / mid_point if mid_point > 0 else 0
                second_half_component = sum(component_scores[mid_point:]) / (len(component_scores) - mid_point) if len(component_scores) - mid_point > 0 else 0
                
                if second_half_component > first_half_component + 0.1:
                    component_trends[component] = 'increasing'
                elif second_half_component < first_half_component - 0.1:
                    component_trends[component] = 'decreasing'
                else:
                    component_trends[component] = 'stable'
        
        return {
            'trend': trend,
            'component_trends': component_trends,
            'data_points': len(recent_assessments),
            'time_window': time_window,
            'first_half_avg': first_half_avg,
            'second_half_avg': second_half_avg
        }
    
    def get_statistics(self) -> Dict:
        """Получение статистики риска"""
        
        if not self.risk_history:
            return {}
        
        # Общая статистика
        total_assessments = len(self.risk_history)
        alert_assessments = [a for a in self.risk_history if a.risk_level == RiskLevel.ALERT]
        monitor_assessments = [a for a in self.risk_history if a.risk_level == RiskLevel.MONITOR]
        normal_assessments = [a for a in self.risk_history if a.risk_level == RiskLevel.NORMAL]
        
        # Средний риск
        avg_risk_score = sum(a.risk_score for a in self.risk_history) / total_assessments
        
        # Максимальный риск
        max_risk_score = max(a.risk_score for a in self.risk_history)
        
        # Распределение уровней риска
        risk_distribution = {
            'normal': len(normal_assessments),
            'monitor': len(monitor_assessments),
            'alert': len(alert_assessments)
        }
        
        # Статистика компонентов
        component_stats = {}
        for component in self.weights.keys():
            component_values = [
                a.component_scores.get(component, 0.0) for a in self.risk_history
            ]
            component_stats[component] = {
                'average': sum(component_values) / len(component_values),
                'max': max(component_values),
                'activation_rate': sum(1 for v in component_values if v > 0.5) / len(component_values)
            }
        
        return {
            'total_assessments': total_assessments,
            'alert_count': len(alert_assessments),
            'monitor_count': len(monitor_assessments),
            'normal_count': len(normal_assessments),
            'alert_rate': len(alert_assessments) / total_assessments,
            'monitor_rate': len(monitor_assessments) / total_assessments,
            'normal_rate': len(normal_assessments) / total_assessments,
            'average_risk_score': avg_risk_score,
            'max_risk_score': max_risk_score,
            'risk_distribution': risk_distribution,
            'component_statistics': component_stats,
            'weights': self.weights,
            'thresholds': {level.value: threshold for level, threshold in self.thresholds.items()}
        }
    
    def should_generate_event(self, assessment: RiskAssessment) -> bool:
        """Проверка, нужно ли генерировать событие"""
        return assessment.risk_level == RiskLevel.ALERT
    
    def get_risk_explanation(self, assessment: RiskAssessment) -> Dict:
        """Получение объяснения оценки риска"""
        
        explanation = {
            'risk_score': assessment.risk_score,
            'risk_level': assessment.risk_level.value,
            'formula': "RiskScore = 0.35*convoy + 0.2*speed + 0.15*direction + 0.15*density + 0.15*thermal",
            'component_breakdown': {},
            'primary_factors': assessment.primary_factors,
            'recommendations': []
        }
        
        # Детальный breakdown по формуле
        for component, weight in self.weights.items():
            component_score = assessment.component_scores.get(component, 0.0)
            contribution = component_score * weight
            explanation['component_breakdown'][component] = {
                'score': component_score,
                'weight': weight,
                'contribution': contribution,
                'percentage': (contribution / assessment.risk_score * 100) if assessment.risk_score > 0 else 0
            }
        
        # Рекомендации на основе ключевых факторов
        recommendations = []
        for factor in assessment.primary_factors:
            if factor == 'convoy_flag':
                recommendations.append("Monitor vehicle convoy formation")
            elif factor == 'anomaly_speed':
                recommendations.append("Check for high-speed vehicle movement")
            elif factor == 'anomaly_direction':
                recommendations.append("Investigate unusual direction changes")
            elif factor == 'density':
                recommendations.append("Monitor high object density areas")
            elif factor == 'thermal_activity':
                recommendations.append("Verify thermal anomalies with ground truth")
        
        explanation['recommendations'] = recommendations
        
        return explanation
    
    def export_risk_data(self, filename: str = None) -> str:
        """Экспорт данных о риске"""
        
        if filename is None:
            filename = f"risk_data_{int(time.time())}.json"
        
        export_data = {
            'metadata': {
                'export_timestamp': time.time(),
                'total_assessments': len(self.risk_history),
                'weights': self.weights,
                'thresholds': {level.value: threshold for level, threshold in self.thresholds.items()}
            },
            'risk_history': [
                {
                    'risk_score': assessment.risk_score,
                    'risk_level': assessment.risk_level.value,
                    'component_scores': assessment.component_scores,
                    'primary_factors': assessment.primary_factors,
                    'timestamp': assessment.timestamp
                }
                for assessment in self.risk_history
            ],
            'statistics': self.get_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename

if __name__ == "__main__":
    # Тестирование Risk Engine
    from fusion import FusedFeatures
    
    risk_engine = RiskEngine()
    
    print("Testing Risk Engine...")
    print("Formula: RiskScore = 0.35*convoy + 0.2*speed + 0.15*direction + 0.15*density + 0.15*thermal")
    
    # Тестовые сценарии
    test_scenarios = [
        {
            'name': 'Normal Scenario',
            'features': FusedFeatures(
                convoy_flag=False, convoy_size=0, anomaly_speed=False,
                anomaly_direction=False, density=0.005, thermal_activity=False,
                thermal_data_available=True, timestamp=time.time()
            )
        },
        {
            'name': 'Convoy Detected',
            'features': FusedFeatures(
                convoy_flag=True, convoy_size=4, anomaly_speed=False,
                anomaly_direction=False, density=0.01, thermal_activity=False,
                thermal_data_available=True, timestamp=time.time()
            )
        },
        {
            'name': 'High Risk Scenario',
            'features': FusedFeatures(
                convoy_flag=True, convoy_size=5, anomaly_speed=True,
                anomaly_direction=True, density=0.02, thermal_activity=True,
                thermal_data_available=True, timestamp=time.time()
            )
        }
    ]
    
    try:
        for scenario in test_scenarios:
            print(f"\n--- {scenario['name']} ---")
            
            # Расчет риска
            assessment = risk_engine.calculate_risk(scenario['features'])
            
            print(f"Risk Score: {assessment.risk_score:.3f}")
            print(f"Risk Level: {assessment.risk_level.value}")
            print(f"Primary Factors: {assessment.primary_factors}")
            
            # Объяснение
            explanation = risk_engine.get_risk_explanation(assessment)
            print(f"Component Breakdown:")
            for component, breakdown in explanation['component_breakdown'].items():
                print(f"  {component}: {breakdown['score']:.2f} * {breakdown['weight']:.2f} = {breakdown['contribution']:.3f} ({breakdown['percentage']:.1f}%)")
            
            print(f"Recommendations: {explanation['recommendations']}")
        
        # Статистика
        stats = risk_engine.get_statistics()
        print(f"\nOverall Statistics:")
        print(f"Total Assessments: {stats['total_assessments']}")
        print(f"Alert Rate: {stats['alert_rate']:.2%}")
        print(f"Average Risk Score: {stats['average_risk_score']:.3f}")
        
        # Тренды
        trends = risk_engine.get_risk_trends()
        print(f"\nRisk Trends: {trends}")
        
        # Экспорт данных
        export_file = risk_engine.export_risk_data()
        print(f"\nData exported to: {export_file}")
        
        print("\nRisk Engine test completed successfully!")
        
    except Exception as e:
        print(f"Error during risk engine test: {e}")
        import traceback
        traceback.print_exc()
