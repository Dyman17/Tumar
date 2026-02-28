from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import json
import asyncio
import uvicorn
from datetime import datetime
import numpy as np
from pedestrian_tracker import PedestrianTracker
import cv2
import base64
from collections import defaultdict, deque
import threading
import time

app = FastAPI(title="Pedestrian Tracking API")

# Глобальные переменные
tracker = None
camera_thread = None
running = False
latest_results = None
frame_buffer = deque(maxlen=1)

class CameraProcessor:
    def __init__(self):
        self.tracker = PedestrianTracker()
        self.cap = None
        self.running = False
        self.frame_count = 0
        
    def start_camera(self, camera_id=0):
        """Запуск камеры"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception("Не удалось открыть камеру")
        
        self.running = True
        self.frame_count = 0
        
    def stop_camera(self):
        """Остановка камеры"""
        self.running = False
        if self.cap:
            self.cap.release()
            
    def process_frames(self):
        """Обработка кадров в отдельном потоке"""
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Обработка кадра
            results = self.tracker.process_frame(frame, self.frame_count)
            
            # Визуализация
            vis_frame = self.tracker.draw_visualization(frame, results)
            
            # Кодирование для веб-трансляции
            _, buffer = cv2.imencode('.jpg', vis_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Сохранение результатов
            global latest_results, frame_buffer
            latest_results = {
                **results,
                'frame_image': frame_base64,
                'timestamp': datetime.now().isoformat()
            }
            frame_buffer.append(frame_base64)
            
            self.frame_count += 1
            
            # Небольшая задержка для управления FPS
            time.sleep(0.03)  # ~30 FPS

camera_processor = CameraProcessor()

@app.get("/")
async def get_index():
    """Главная страница с визуализацией"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Pedestrian Tracking Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: white; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .video-section { display: flex; gap: 20px; margin-bottom: 30px; }
        .video-container { flex: 1; }
        .stats-container { width: 300px; }
        .video-stream { width: 100%; border-radius: 10px; }
        .heatmap-container { margin-bottom: 30px; }
        .chart-container { background: #2a2a2a; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .controls { text-align: center; margin-bottom: 20px; }
        button { padding: 10px 20px; margin: 0 10px; border: none; border-radius: 5px; cursor: pointer; }
        .start-btn { background: #4CAF50; color: white; }
        .stop-btn { background: #f44336; color: white; }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .stat-card { background: #2a2a2a; padding: 15px; border-radius: 10px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #4CAF50; }
        .stat-label { color: #ccc; margin-top: 5px; }
        canvas { max-height: 300px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚶‍♂️ Pedestrian Tracking Dashboard</h1>
            <p>Real-time pedestrian detection, tracking and trajectory prediction</p>
        </div>
        
        <div class="controls">
            <button class="start-btn" onclick="startTracking()">Start Tracking</button>
            <button class="stop-btn" onclick="stopTracking()">Stop Tracking</button>
        </div>
        
        <div class="video-section">
            <div class="video-container">
                <h3>Live Video Stream</h3>
                <img id="videoStream" class="video-stream" src="" alt="Video stream will appear here">
            </div>
            
            <div class="stats-container">
                <h3>Real-time Statistics</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="currentCount">0</div>
                        <div class="stat-label">Current People</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="avgCount">0</div>
                        <div class="stat-label">Average (10s)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="activeTracks">0</div>
                        <div class="stat-label">Active Tracks</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="predictions">0</div>
                        <div class="stat-label">Predictions</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <canvas id="densityChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="heatmap-container">
            <h3>Density Heatmap</h3>
            <div class="chart-container">
                <canvas id="heatmapCanvas" width="800" height="400"></canvas>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Trajectory Analysis</h3>
            <canvas id="trajectoryChart"></canvas>
        </div>
    </div>

    <script>
        let ws;
        let densityChart;
        let trajectoryChart;
        let densityData = [];
        let trajectoryData = [];
        
        // Инициализация графиков
        function initCharts() {
            // График плотности
            const densityCtx = document.getElementById('densityChart').getContext('2d');
            densityChart = new Chart(densityCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'People Count',
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
                        y: { beginAtZero: true }
                    }
                }
            });
            
            // График траекторий
            const trajectoryCtx = document.getElementById('trajectoryChart').getContext('2d');
            trajectoryChart = new Chart(trajectoryCtx, {
                type: 'scatter',
                data: {
                    datasets: []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'X Position' } },
                        y: { title: { display: true, text: 'Y Position' } }
                    }
                }
            });
        }
        
        // WebSocket соединение
        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onclose = function() {
                setTimeout(connectWebSocket, 3000);
            };
        }
        
        // Обновление дашборда
        function updateDashboard(data) {
            // Обновление видео
            if (data.frame_image) {
                document.getElementById('videoStream').src = 'data:image/jpeg;base64,' + data.frame_image;
            }
            
            // Обновление статистики
            if (data.density) {
                document.getElementById('currentCount').textContent = data.density.current;
                document.getElementById('avgCount').textContent = data.density.average_10s.toFixed(1);
            }
            
            document.getElementById('activeTracks').textContent = Object.keys(data.tracks || {}).length;
            document.getElementById('predictions').textContent = Object.keys(data.predictions || {}).length;
            
            // Обновление графика плотности
            if (data.density) {
                const now = new Date().toLocaleTimeString();
                densityData.push({
                    time: now,
                    count: data.density.current
                });
                
                if (densityData.length > 20) densityData.shift();
                
                densityChart.data.labels = densityData.map(d => d.time);
                densityChart.data.datasets[0].data = densityData.map(d => d.count);
                densityChart.update('none');
            }
            
            // Обновление heatmap
            updateHeatmap(data.heatmap_data || []);
        }
        
        // Обновление heatmap
        function updateHeatmap(points) {
            const canvas = document.getElementById('heatmapCanvas');
            const ctx = canvas.getContext('2d');
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Простая визуализация точек
            points.forEach(point => {
                const x = (point.x / 640) * canvas.width;
                const y = (point.y / 480) * canvas.height;
                
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
                ctx.fill();
            });
        }
        
        // Управление трекингом
        async function startTracking() {
            try {
                const response = await fetch('/start', { method: 'POST' });
                const result = await response.json();
                console.log(result.message);
            } catch (error) {
                console.error('Error starting tracking:', error);
            }
        }
        
        async function stopTracking() {
            try {
                const response = await fetch('/stop', { method: 'POST' });
                const result = await response.json();
                console.log(result.message);
            } catch (error) {
                console.error('Error stopping tracking:', error);
            }
        }
        
        // Инициализация
        window.onload = function() {
            initCharts();
            connectWebSocket();
        };
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/start")
async def start_tracking():
    """Запуск трекинга"""
    global camera_thread, running
    
    try:
        camera_processor.start_camera(0)  # Камера по умолчанию
        
        # Запуск обработки в отдельном потоке
        camera_thread = threading.Thread(target=camera_processor.process_frames)
        camera_thread.daemon = True
        camera_thread.start()
        
        running = True
        
        return {"message": "Tracking started successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/stop")
async def stop_tracking():
    """Остановка трекинга"""
    global running
    
    camera_processor.stop_camera()
    running = False
    
    return {"message": "Tracking stopped"}

@app.get("/status")
async def get_status():
    """Получение текущего статуса"""
    return {
        "running": running,
        "timestamp": datetime.now().isoformat(),
        "results": latest_results
    }

@app.get("/heatmap")
async def get_heatmap():
    """Получение данных для heatmap"""
    if not camera_processor.tracker:
        return {"points": []}
    
    heatmap_data = camera_processor.tracker.get_heatmap_data()
    return {"points": heatmap_data}

@app.get("/statistics")
async def get_statistics():
    """Получение статистики"""
    if not latest_results:
        return {"error": "No data available"}
    
    return {
        "density": latest_results.get("density", {}),
        "tracks_count": len(latest_results.get("tracks", {})),
        "predictions_count": len(latest_results.get("predictions", {})),
        "frame_idx": latest_results.get("frame_idx", 0),
        "timestamp": latest_results.get("timestamp")
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket для real-time обновлений"""
    await websocket.accept()
    
    try:
        while True:
            if latest_results:
                # Отправляем последние результаты
                await websocket.send_json(latest_results)
            
            await asyncio.sleep(0.1)  # 10 FPS для WebSocket
            
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
