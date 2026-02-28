from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import uvicorn
import time
import json
import cv2
import numpy as np
import base64
import io
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import threading
import queue

# Импорты наших AI модулей
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai_core'))

from data_fusion import DataFusionEngine, FusedData
from risk_engine import RiskEngine, RiskAssessment, Alert
from object_detection import DetectionSystem
from anomaly_detection import AnomalyDetectionSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic модели для API
class LocationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius: float = Field(default=0.1, gt=0)

class SensorDataRequest(BaseModel):
    timestamp: float
    temperature: float
    humidity: float
    light_level: float
    pressure: Optional[float] = None
    wind_speed: Optional[float] = None
    gps: Optional[Dict[str, float]] = None

class RiskQuery(BaseModel):
    location: Tuple[float, float]
    time_range: Optional[Tuple[float, float]] = None
    risk_level: Optional[str] = None

class AlertResponse(BaseModel):
    alert_id: str
    alert_type: str
    severity: str
    title: str
    description: str
    risk_score: float
    location: Optional[Tuple[float, float]]
    timestamp: float
    is_active: bool

class RiskResponse(BaseModel):
    total_score: float
    risk_level: str
    confidence: float
    primary_factors: List[str]
    secondary_factors: List[str]
    recommendations: List[str]
    timestamp: float

class DetectionResponse(BaseModel):
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    track_id: Optional[int]

class FrameAnalysisResponse(BaseModel):
    frame_id: int
    timestamp: float
    detections: List[DetectionResponse]
    risk_assessment: RiskResponse
    processing_time: float

# Глобальные переменные
app = FastAPI(
    title="Distributed AI Monitoring API",
    description="API for distributed AI satellite monitoring system",
    version="1.0.0"
)

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация систем
fusion_engine = DataFusionEngine()
risk_engine = RiskEngine()
detection_system = DetectionSystem()

# Очереди и буферы
frame_queue = queue.Queue(maxsize=100)
result_queue = queue.Queue(maxsize=100)
websocket_connections = set()

# История данных
processing_history = deque(maxlen=1000)
api_statistics = defaultdict(int)

class StreamingProcessor:
    """Процессор для стриминга видео"""
    
    def __init__(self):
        self.is_running = False
        self.camera = None
        self.processing_thread = None
        self.frame_count = 0
        
    def start_camera(self, camera_id: int = 0):
        """Запуск камеры"""
        try:
            self.camera = cv2.VideoCapture(camera_id)
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {camera_id}")
            
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_stream)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            logger.info(f"Camera {camera_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Остановка камеры"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("Camera stopped")
    
    def _process_stream(self):
        """Обработка видеопотока"""
        while self.is_running and self.camera:
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            self.frame_count += 1
            
            try:
                # Обработка кадра
                start_time = time.time()
                
                # Fusion обработка
                fused_data = fusion_engine.fuse_frame_data(frame)
                
                # Оценка риска
                risk_assessment = risk_engine.assess_comprehensive_risk(fused_data)
                
                processing_time = time.time() - start_time
                
                # Формируем результат
                result = {
                    'frame_id': self.frame_count,
                    'timestamp': time.time(),
                    'detections': [
                        {
                            'class_name': det.class_name,
                            'confidence': det.confidence,
                            'bbox': det.bbox,
                            'center': det.center,
                            'track_id': det.track_id
                        }
                        for det in fused_data.local_detections
                    ],
                    'risk_assessment': {
                        'total_score': risk_assessment.total_score,
                        'risk_level': risk_assessment.risk_level.value,
                        'confidence': risk_assessment.confidence,
                        'primary_factors': risk_assessment.primary_factors,
                        'secondary_factors': risk_assessment.secondary_factors,
                        'recommendations': risk_assessment.recommendations,
                        'timestamp': risk_assessment.timestamp
                    },
                    'processing_time': processing_time,
                    'frame_image': self._encode_frame(frame, fused_data, risk_assessment)
                }
                
                # Отправляем в очередь
                if not result_queue.full():
                    result_queue.put(result)
                
                # Обновляем статистику
                processing_history.append(result)
                
                # WebSocket рассылка
                asyncio.create_task(self._broadcast_result(result))
                
            except Exception as e:
                logger.error(f"Error processing frame {self.frame_count}: {e}")
    
    def _encode_frame(self, frame: np.ndarray, fused_data: FusedData, 
                    risk_assessment: RiskAssessment) -> str:
        """Кодирование кадра с overlay"""
        
        # Отрисовка детекций
        vis_frame = detection_system.detector.draw_detections(frame, fused_data.local_detections)
        
        # Добавляем информацию о риске
        risk_text = f"Risk: {risk_assessment.risk_level.value} ({risk_assessment.total_score:.2f})"
        cv2.putText(vis_frame, risk_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Добавляем количество детекций
        detection_text = f"Objects: {len(fused_data.local_detections)}"
        cv2.putText(vis_frame, detection_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Кодируем в base64
        _, buffer = cv2.imencode('.jpg', vis_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return frame_base64
    
    async def _broadcast_result(self, result: Dict):
        """Рассылка результата по WebSocket"""
        if websocket_connections:
            message = json.dumps(result)
            disconnected = set()
            
            for websocket in websocket_connections:
                try:
                    await websocket.send_text(message)
                except:
                    disconnected.add(websocket)
            
            # Удаляем отключенные соединения
            websocket_connections.difference_update(disconnected)

streaming_processor = StreamingProcessor()

# API Endpoints
@app.get("/")
async def get_root():
    """Корневой endpoint с документацией"""
    return {
        "message": "Distributed AI Monitoring API",
        "version": "1.0.0",
        "endpoints": {
            "streaming": "/api/v1/streaming",
            "analysis": "/api/v1/analyze",
            "risk": "/api/v1/risk",
            "alerts": "/api/v1/alerts",
            "statistics": "/api/v1/statistics",
            "health": "/api/v1/health"
        }
    }

@app.get("/api/v1/health")
async def health_check():
    """Проверка здоровья системы"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "systems": {
            "fusion_engine": "operational",
            "risk_engine": "operational",
            "detection_system": "operational",
            "streaming": "active" if streaming_processor.is_running else "inactive"
        }
    }

@app.post("/api/v1/streaming/start")
async def start_streaming(camera_id: int = 0):
    """Запуск стриминга"""
    success = streaming_processor.start_camera(camera_id)
    
    if success:
        return {
            "message": "Streaming started successfully",
            "camera_id": camera_id,
            "timestamp": time.time()
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to start streaming")

@app.post("/api/v1/streaming/stop")
async def stop_streaming():
    """Остановка стриминга"""
    streaming_processor.stop_camera()
    
    return {
        "message": "Streaming stopped",
        "timestamp": time.time()
    }

@app.get("/api/v1/streaming/status")
async def get_streaming_status():
    """Статус стриминга"""
    return {
        "is_running": streaming_processor.is_running,
        "frame_count": streaming_processor.frame_count,
        "queue_size": result_queue.qsize(),
        "active_websockets": len(websocket_connections)
    }

@app.post("/api/v1/analyze/frame")
async def analyze_frame(background_tasks: BackgroundTasks, 
                       sensor_data: Optional[SensorDataRequest] = None):
    """Анализ одного кадра"""
    
    if not streaming_processor.camera:
        raise HTTPException(status_code=400, detail="Camera not started")
    
    # Получаем кадр
    ret, frame = streaming_processor.camera.read()
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to capture frame")
    
    try:
        # Обработка кадра
        start_time = time.time()
        
        sensor_dict = sensor_data.dict() if sensor_data else None
        fused_data = fusion_engine.fuse_frame_data(frame, sensor_dict)
        risk_assessment = risk_engine.assess_comprehensive_risk(fused_data)
        
        processing_time = time.time() - start_time
        
        # Формируем ответ
        response = FrameAnalysisResponse(
            frame_id=streaming_processor.frame_count,
            timestamp=time.time(),
            detections=[
                DetectionResponse(
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    center=det.center,
                    track_id=det.track_id
                )
                for det in fused_data.local_detections
            ],
            risk_assessment=RiskResponse(
                total_score=risk_assessment.total_score,
                risk_level=risk_assessment.risk_level.value,
                confidence=risk_assessment.confidence,
                primary_factors=risk_assessment.primary_factors,
                secondary_factors=risk_assessment.secondary_factors,
                recommendations=risk_assessment.recommendations,
                timestamp=risk_assessment.timestamp
            ),
            processing_time=processing_time
        )
        
        # Обновляем статистику
        api_statistics['frame_analyses'] += 1
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing frame: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/v1/risk/current")
async def get_current_risk():
    """Получение текущей оценки риска"""
    
    if not risk_engine.assessment_history:
        raise HTTPException(status_code=404, detail="No risk assessments available")
    
    latest_assessment = list(risk_engine.assessment_history)[-1]
    
    return RiskResponse(
        total_score=latest_assessment.total_score,
        risk_level=latest_assessment.risk_level.value,
        confidence=latest_assessment.confidence,
        primary_factors=latest_assessment.primary_factors,
        secondary_factors=latest_assessment.secondary_factors,
        recommendations=latest_assessment.recommendations,
        timestamp=latest_assessment.timestamp
    )

@app.post("/api/v1/risk/assess")
async def assess_risk(location: LocationRequest, 
                      sensor_data: Optional[SensorDataRequest] = None):
    """Оценка риска для локации"""
    
    try:
        # Обновляем локацию
        fusion_engine.location = (location.latitude, location.longitude)
        risk_engine.location = (location.latitude, location.longitude)
        
        # Создаем тестовый кадр (или получаем реальный)
        if streaming_processor.camera:
            ret, frame = streaming_processor.camera.read()
            if ret:
                sensor_dict = sensor_data.dict() if sensor_data else None
                fused_data = fusion_engine.fuse_frame_data(frame, sensor_dict)
            else:
                raise HTTPException(status_code=500, detail="Failed to capture frame")
        else:
            # Симуляция без камеры
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            sensor_dict = sensor_data.dict() if sensor_data else None
            fused_data = fusion_engine.fuse_frame_data(test_frame, sensor_dict)
        
        # Оценка риска
        risk_assessment = risk_engine.assess_comprehensive_risk(fused_data)
        
        api_statistics['risk_assessments'] += 1
        
        return RiskResponse(
            total_score=risk_assessment.total_score,
            risk_level=risk_assessment.risk_level.value,
            confidence=risk_assessment.confidence,
            primary_factors=risk_assessment.primary_factors,
            secondary_factors=risk_assessment.secondary_factors,
            recommendations=risk_assessment.recommendations,
            timestamp=risk_assessment.timestamp
        )
        
    except Exception as e:
        logger.error(f"Error assessing risk: {e}")
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

@app.get("/api/v1/risk/trends")
async def get_risk_trends(hours: int = 24):
    """Получение трендов риска"""
    
    trends = risk_engine.get_risk_trends(hours)
    
    return {
        "timeframe_hours": hours,
        "trends": trends,
        "timestamp": time.time()
    }

@app.get("/api/v1/alerts")
async def get_alerts(active_only: bool = True):
    """Получение алертов"""
    
    if active_only:
        alerts = risk_engine.get_active_alerts()
    else:
        alerts = list(risk_engine.alert_history)
    
    return [
        AlertResponse(
            alert_id=alert.alert_id,
            alert_type=alert.alert_type.value,
            severity=alert.severity,
            title=alert.title,
            description=alert.description,
            risk_score=alert.risk_score,
            location=alert.location,
            timestamp=alert.timestamp,
            is_active=alert.is_active
        )
        for alert in alerts
    ]

@app.post("/api/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Подтверждение алерта"""
    
    success = risk_engine.acknowledge_alert(alert_id)
    
    if success:
        return {"message": f"Alert {alert_id} acknowledged"}
    else:
        raise HTTPException(status_code=404, detail="Alert not found")

@app.get("/api/v1/statistics")
async def get_statistics():
    """Получение статистики системы"""
    
    fusion_stats = fusion_engine.get_fusion_statistics()
    risk_stats = risk_engine.get_risk_statistics()
    
    return {
        "timestamp": time.time(),
        "fusion_engine": fusion_stats,
        "risk_engine": risk_stats,
        "api_statistics": dict(api_statistics),
        "streaming": {
            "is_running": streaming_processor.is_running,
            "frame_count": streaming_processor.frame_count,
            "queue_size": result_queue.qsize()
        }
    }

@app.get("/api/v1/stream")
async def get_processed_stream():
    """Получение обработанного видеопотока"""
    
    if not streaming_processor.is_running:
        raise HTTPException(status_code=400, detail="Streaming not started")
    
    async def generate_stream():
        while streaming_processor.is_running:
            try:
                if not result_queue.empty():
                    result = result_queue.get()
                    
                    # Отправляем кадр как MJPEG
                    frame_data = base64.b64decode(result['frame_image'])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                else:
                    await asyncio.sleep(0.033)  # ~30 FPS
                    
            except Exception as e:
                logger.error(f"Error in stream generation: {e}")
                break
    
    return StreamingResponse(
        generate_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket для real-time обновлений"""
    
    await websocket.accept()
    websocket_connections.add(websocket)
    
    try:
        while True:
            # Отправляем последние результаты
            if not result_queue.empty():
                result = result_queue.queue[0]  # Берем последний результат без удаления
                await websocket.send_json(result)
            
            await asyncio.sleep(0.1)  # 10 FPS для WebSocket
            
    except WebSocketDisconnect:
        websocket_connections.discard(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_connections.discard(websocket)

# Статические файлы для фронтенда
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/dashboard")
async def get_dashboard():
    """Веб-дашборд"""
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: white; }
        .container { max-width: 1600px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        .card { background: #2a2a2a; padding: 20px; border-radius: 10px; }
        .video-container { grid-column: span 2; }
        .stream-viewer { width: 100%; border-radius: 10px; }
        .risk-meter { font-size: 3em; font-weight: bold; text-align: center; }
        .risk-low { color: #4CAF50; }
        .risk-medium { color: #FF9800; }
        .risk-high { color: #f44336; }
        .risk-critical { color: #9C27B0; }
        .map-container { height: 400px; border-radius: 10px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .stat-item { text-align: center; padding: 15px; background: #333; border-radius: 8px; }
        .stat-value { font-size: 2em; font-weight: bold; color: #4CAF50; }
        .alert-item { padding: 10px; margin: 5px 0; border-radius: 5px; }
        .alert-critical { background: #f44336; }
        .alert-high { background: #FF9800; }
        .alert-medium { background: #2196F3; }
        .alert-low { background: #4CAF50; }
        button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
        .btn-start { background: #4CAF50; color: white; }
        .btn-stop { background: #f44336; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛰 Distributed AI Monitoring System</h1>
            <p>Real-time risk assessment and anomaly detection</p>
        </div>
        
        <div class="controls">
            <button class="btn-start" onclick="startStreaming()">Start Streaming</button>
            <button class="btn-stop" onclick="stopStreaming()">Stop Streaming</button>
            <button onclick="refreshData()">Refresh Data</button>
        </div>
        
        <div class="grid">
            <div class="card video-container">
                <h3>Live Video Stream</h3>
                <img id="videoStream" class="stream-viewer" src="" alt="Video stream">
            </div>
            
            <div class="card">
                <h3>Risk Assessment</h3>
                <div id="riskMeter" class="risk-meter risk-low">LOW</div>
                <div id="riskScore">Score: 0.00</div>
                <div id="riskFactors"></div>
            </div>
        </div>
        
        <div class="card">
            <h3>System Statistics</h3>
            <div class="stats-grid" id="statsGrid">
                <!-- Статистика будет загружена через JS -->
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>Active Alerts</h3>
                <div id="alertsList">
                    <!-- Алерты будут загружены через JS -->
                </div>
            </div>
            
            <div class="card">
                <h3>Location Map</h3>
                <div id="map" class="map-container"></div>
            </div>
        </div>
        
        <div class="card">
            <h3>Risk Trends</h3>
            <canvas id="riskChart"></canvas>
        </div>
    </div>

    <script>
        let ws;
        let riskChart;
        let map;
        
        // Инициализация
        window.onload = function() {
            initMap();
            initChart();
            connectWebSocket();
            refreshData();
        };
        
        function initMap() {
            map = L.map('map').setView([55.7558, 37.6173], 13);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(map);
        }
        
        function initChart() {
            const ctx = document.getElementById('riskChart').getContext('2d');
            riskChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Risk Score',
                        data: [],
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { beginAtZero: true, max: 1 }
                    }
                }
            });
        }
        
        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:8000/ws/realtime');
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onclose = function() {
                setTimeout(connectWebSocket, 3000);
            };
        }
        
        function updateDashboard(data) {
            // Обновляем видео
            if (data.frame_image) {
                document.getElementById('videoStream').src = 'data:image/jpeg;base64,' + data.frame_image;
            }
            
            // Обновляем риск
            if (data.risk_assessment) {
                updateRiskDisplay(data.risk_assessment);
            }
        }
        
        function updateRiskDisplay(riskData) {
            const riskMeter = document.getElementById('riskMeter');
            const riskScore = document.getElementById('riskScore');
            const riskFactors = document.getElementById('riskFactors');
            
            // Обновляем уровень риска
            riskMeter.textContent = riskData.risk_level;
            riskScore.textContent = `Score: ${riskData.total_score.toFixed(2)}`;
            
            // Обновляем цвет
            riskMeter.className = 'risk-meter';
            switch(riskData.risk_level) {
                case 'LOW':
                    riskMeter.classList.add('risk-low');
                    break;
                case 'MEDIUM':
                    riskMeter.classList.add('risk-medium');
                    break;
                case 'HIGH':
                    riskMeter.classList.add('risk-high');
                    break;
                case 'CRITICAL':
                    riskMeter.classList.add('risk-critical');
                    break;
            }
            
            // Обновляем факторы
            riskFactors.innerHTML = '<strong>Primary Factors:</strong><br>' + 
                riskData.primary_factors.join(', ');
            
            // Обновляем график
            updateRiskChart(riskData.total_score);
        }
        
        function updateRiskChart(score) {
            const now = new Date().toLocaleTimeString();
            
            if (riskChart.data.labels.length > 20) {
                riskChart.data.labels.shift();
                riskChart.data.datasets[0].data.shift();
            }
            
            riskChart.data.labels.push(now);
            riskChart.data.datasets[0].data.push(score);
            riskChart.update('none');
        }
        
        async function startStreaming() {
            try {
                const response = await fetch('/api/v1/streaming/start', { method: 'POST' });
                const result = await response.json();
                console.log(result.message);
                
                // Начинаем показ видео
                document.getElementById('videoStream').src = '/api/v1/stream';
                
            } catch (error) {
                console.error('Error starting streaming:', error);
            }
        }
        
        async function stopStreaming() {
            try {
                const response = await fetch('/api/v1/streaming/stop', { method: 'POST' });
                const result = await response.json();
                console.log(result.message);
                
                // Останавливаем видео
                document.getElementById('videoStream').src = '';
                
            } catch (error) {
                console.error('Error stopping streaming:', error);
            }
        }
        
        async function refreshData() {
            try {
                // Загружаем статистику
                const statsResponse = await fetch('/api/v1/statistics');
                const stats = await statsResponse.json();
                updateStatistics(stats);
                
                // Загружаем алерты
                const alertsResponse = await fetch('/api/v1/alerts');
                const alerts = await alertsResponse.json();
                updateAlerts(alerts);
                
            } catch (error) {
                console.error('Error refreshing data:', error);
            }
        }
        
        function updateStatistics(stats) {
            const statsGrid = document.getElementById('statsGrid');
            
            statsGrid.innerHTML = `
                <div class="stat-item">
                    <div class="stat-value">${stats.risk_engine.current_risk?.toFixed(2) || '0.00'}</div>
                    <div>Current Risk</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.risk_engine.total_assessments || 0}</div>
                    <div>Total Assessments</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.risk_engine.active_alerts || 0}</div>
                    <div>Active Alerts</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.streaming.frame_count || 0}</div>
                    <div>Frames Processed</div>
                </div>
            `;
        }
        
        function updateAlerts(alerts) {
            const alertsList = document.getElementById('alertsList');
            
            if (alerts.length === 0) {
                alertsList.innerHTML = '<p>No active alerts</p>';
                return;
            }
            
            alertsList.innerHTML = alerts.map(alert => `
                <div class="alert-item alert-${alert.severity.toLowerCase()}">
                    <strong>${alert.title}</strong><br>
                    ${alert.description}<br>
                    <small>Score: ${alert.risk_score.toFixed(2)}</small>
                </div>
            `).join('');
        }
        
        // Автообновление каждые 30 секунд
        setInterval(refreshData, 30000);
    </script>
</body>
</html>
    """
    
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    print("Starting Distributed AI Monitoring API Server...")
    print("Dashboard available at: http://localhost:8000/dashboard")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
