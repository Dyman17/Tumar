"""
Cloud API - Облачный сервер для деплоя на Render/Vercel
Принимает данные от локального ноутбука и отдает веб-интерфейс
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from typing import Dict, List
import time
from datetime import datetime

app = FastAPI(title="AI Monitor Cloud API", version="1.0.0")

# CORS для локального подключения
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Хранилище данных (в проде - Redis/PostgreSQL)
current_data = {
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
    'fps': 0.0,
    'connection_status': 'offline'
}

# WebSocket клиенты (веб-интерфейсы)
web_clients: List[WebSocket] = []
# Локальный ноутбук (источник данных)
laptop_client: WebSocket = None

# История для графиков
risk_history = []
max_history = 100

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Главная страница с дашбордом"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AI Monitor Cloud</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #00ff88, #00bbff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255,255,255,0.1);
            padding: 15px 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .status-online { background: #4CAF50; }
        .status-offline { background: #f44336; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .video-container {
            text-align: center;
        }
        
        .video-stream {
            width: 100%;
            max-width: 400px;
            border-radius: 10px;
            border: 2px solid rgba(255,255,255,0.3);
        }
        
        .risk-container h3 {
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .risk-bar {
            width: 100%;
            height: 30px;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            overflow: hidden;
            margin: 15px 0;
        }
        
        .risk-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #FFC107, #f44336);
            transition: width 0.5s ease;
            border-radius: 15px;
        }
        
        .risk-score {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            margin: 15px 0;
        }
        
        .risk-normal { color: #4CAF50; }
        .risk-monitor { color: #FFC107; }
        .risk-alert { color: #f44336; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
        }
        
        .stat-item {
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #00ff88;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .alert-banner {
            background: rgba(244, 67, 54, 0.2);
            border: 1px solid #f44336;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            display: none;
            animation: alertPulse 1s infinite;
        }
        
        .alert-banner.show {
            display: block;
        }
        
        @keyframes alertPulse {
            0% { background: rgba(244, 67, 54, 0.2); }
            50% { background: rgba(244, 67, 54, 0.4); }
            100% { background: rgba(244, 67, 54, 0.2); }
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            opacity: 0.7;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            font-size: 1.2em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 AI Monitor Cloud</h1>
            <p>Real-time Scene Analysis Dashboard</p>
        </div>
        
        <div class="status-bar">
            <div class="status-indicator">
                <div id="statusDot" class="status-dot status-offline"></div>
                <span id="statusText">Offline</span>
            </div>
            <div>
                <span id="timestamp">Waiting for connection...</span>
            </div>
        </div>
        
        <div id="alertBanner" class="alert-banner">
            <strong>⚠️ ALERT:</strong> <span id="alertMessage">High risk detected</span>
        </div>
        
        <div class="main-grid">
            <div class="card video-container">
                <h3>📹 Live Stream</h3>
                <div id="loadingVideo" class="loading">Waiting for video stream...</div>
                <img id="videoStream" class="video-stream" style="display:none;" src="" alt="AI video stream">
                <div style="margin-top: 10px; font-size: 0.9em; opacity: 0.8;">
                    FPS: <span id="fpsDisplay">0.0</span>
                </div>
            </div>
            
            <div class="card risk-container">
                <h3>🚨 Risk Assessment</h3>
                <div class="risk-bar">
                    <div id="riskFill" class="risk-fill" style="width: 0%"></div>
                </div>
                <div id="riskScore" class="risk-score risk-normal">0.00 - NORMAL</div>
                
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="vehiclesCount">0</div>
                        <div class="stat-label">Vehicles</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="personsCount">0</div>
                        <div class="stat-label">Persons</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="avgSpeed">0.0</div>
                        <div class="stat-label">Speed m/s</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="convoyStatus">NO</div>
                        <div class="stat-label">Convoy</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>📊 Scene Analysis</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" id="thermalStatus">NO</div>
                    <div class="stat-label">Thermal</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="frameId">0</div>
                    <div class="stat-label">Frame ID</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="uptime">0s</div>
                    <div class="stat-label">Uptime</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="dataRate">0 KB/s</div>
                    <div class="stat-label">Data Rate</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>AI Monitor Cloud v1.0 | Real-time Intelligence Platform</p>
        </div>
    </div>

    <script>
        let ws;
        let reconnectInterval;
        let startTime = Date.now();
        let lastDataTime = 0;
        let dataBytes = 0;
        
        function connectWebSocket() {
            ws = new WebSocket('ws://' + window.location.host + '/ws');
            
            ws.onopen = function() {
                console.log('Connected to cloud API');
                updateConnectionStatus(true);
                
                if (reconnectInterval) {
                    clearInterval(reconnectInterval);
                    reconnectInterval = null;
                }
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
                lastDataTime = Date.now();
                dataBytes += event.data.length;
            };
            
            ws.onclose = function() {
                console.log('Disconnected from cloud API');
                updateConnectionStatus(false);
                
                if (!reconnectInterval) {
                    reconnectInterval = setInterval(connectWebSocket, 3000);
                }
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateDashboard(data) {
            // Обновляем видео
            if (data.video_frame) {
                const videoElement = document.getElementById('videoStream');
                const loadingElement = document.getElementById('loadingVideo');
                
                videoElement.src = 'data:image/jpeg;base64,' + data.video_frame;
                videoElement.style.display = 'block';
                loadingElement.style.display = 'none';
            }
            
            // Обновляем риск
            updateRiskDisplay(data.risk_score, data.risk_level);
            
            // Обновляем статистику
            updateStats(data.scene);
            
            // Обновляем технические данные
            document.getElementById('fpsDisplay').textContent = data.fps.toFixed(1);
            document.getElementById('frameId').textContent = data.frame_id || '0';
            
            // Обновляем время
            updateTimestamp();
        }
        
        function updateRiskDisplay(score, level) {
            const riskScore = document.getElementById('riskScore');
            const riskFill = document.getElementById('riskFill');
            const alertBanner = document.getElementById('alertBanner');
            
            // Обновляем текст
            riskScore.textContent = `${score.toFixed(2)} - ${level}`;
            
            // Обновляем цвет
            riskScore.className = 'risk-score';
            if (level === 'NORMAL') {
                riskScore.classList.add('risk-normal');
                alertBanner.classList.remove('show');
            } else if (level === 'MONITOR') {
                riskScore.classList.add('risk-monitor');
                alertBanner.classList.remove('show');
            } else {
                riskScore.classList.add('risk-alert');
                alertBanner.classList.add('show');
            }
            
            // Обновляем полосу
            riskFill.style.width = `${score * 100}%`;
        }
        
        function updateStats(scene) {
            document.getElementById('vehiclesCount').textContent = scene.vehicles || 0;
            document.getElementById('personsCount').textContent = scene.persons || 0;
            document.getElementById('avgSpeed').textContent = (scene.avg_speed || 0).toFixed(1);
            document.getElementById('convoyStatus').textContent = scene.convoy_detected ? 'YES' : 'NO';
            document.getElementById('thermalStatus').textContent = scene.thermal_active ? 'YES' : 'NO';
            
            // Подсветка критических значений
            const convoyElement = document.getElementById('convoyStatus');
            if (scene.convoy_detected) {
                convoyElement.style.color = '#f44336';
            } else {
                convoyElement.style.color = '#00ff88';
            }
            
            const thermalElement = document.getElementById('thermalStatus');
            if (scene.thermal_active) {
                thermalElement.style.color = '#f44336';
            } else {
                thermalElement.style.color = '#00ff88';
            }
        }
        
        function updateConnectionStatus(connected) {
            const indicator = document.getElementById('statusDot');
            const text = document.getElementById('statusText');
            
            if (connected) {
                indicator.className = 'status-dot status-online';
                text.textContent = 'Online';
                current_data.connection_status = 'online';
            } else {
                indicator.className = 'status-dot status-offline';
                text.textContent = 'Offline';
                current_data.connection_status = 'offline';
            }
        }
        
        function updateTimestamp() {
            const now = new Date();
            document.getElementById('timestamp').textContent = now.toLocaleTimeString();
            
            // Uptime
            const uptime = Math.floor((Date.now() - startTime) / 1000);
            document.getElementById('uptime').textContent = uptime + 's';
            
            // Data rate (простой расчет)
            if (lastDataTime > 0) {
                const timeDiff = (Date.now() - lastDataTime) / 1000;
                const dataRate = timeDiff > 0 ? (dataBytes / 1024 / timeDiff).toFixed(1) : '0';
                document.getElementById('dataRate').textContent = dataRate + ' KB/s';
            }
        }
        
        // Запуск
        window.onload = function() {
            connectWebSocket();
            
            // Обновление времени каждую секунду
            setInterval(updateTimestamp, 1000);
        };
        
        // Очистка при закрытии
        window.onbeforeunload = function() {
            if (ws) {
                ws.close();
            }
            if (reconnectInterval) {
                clearInterval(reconnectInterval);
            }
        };
    </script>
</body>
</html>
    """

@app.post("/api/data")
async def receive_data(data: Dict):
    """Принимает данные от локального ноутбука"""
    global current_data, risk_history
    
    # Обновляем текущие данные
    current_data.update(data)
    current_data['timestamp'] = time.time()
    current_data['connection_status'] = 'online'
    
    # Сохраняем историю риска
    if 'risk_score' in data:
        risk_history.append({
            'timestamp': current_data['timestamp'],
            'risk_score': data['risk_score'],
            'risk_level': data.get('risk_level', 'NORMAL')
        })
        
        # Ограничиваем историю
        if len(risk_history) > max_history:
            risk_history.pop(0)
    
    # Рассылаем веб-клиентам
    await broadcast_to_web_clients(data)
    
    return {"status": "received", "timestamp": current_data['timestamp']}

@app.get("/api/status")
async def get_status():
    """Статус системы"""
    return {
        "status": "online",
        "timestamp": current_data['timestamp'],
        "connection_status": current_data['connection_status'],
        "web_clients": len(web_clients),
        "laptop_connected": laptop_client is not None,
        "data_age": time.time() - current_data['timestamp'] if current_data['timestamp'] > 0 else None
    }

@app.get("/api/history")
async def get_history():
    """История данных риска"""
    return {
        "risk_history": risk_history,
        "current_data": current_data
    }

@app.websocket("/ws/laptop")
async def websocket_laptop(websocket: WebSocket):
    """WebSocket для подключения локального ноутбука"""
    global laptop_client
    
    await websocket.accept()
    laptop_client = websocket
    print(f"Laptop connected: {websocket.client}")
    
    try:
        while True:
            # Получаем данные от ноутбука
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Обновляем глобальные данные
            current_data.update(message)
            current_data['timestamp'] = time.time()
            current_data['connection_status'] = 'online'
            
            # Сохраняем историю риска
            if 'risk_score' in message:
                risk_history.append({
                    'timestamp': current_data['timestamp'],
                    'risk_score': message['risk_score'],
                    'risk_level': message.get('risk_level', 'NORMAL')
                })
                
                if len(risk_history) > max_history:
                    risk_history.pop(0)
            
            # Рассылаем веб-клиентам
            await broadcast_to_web_clients(message)
            
    except WebSocketDisconnect:
        print(f"Laptop disconnected: {websocket.client}")
        laptop_client = None
        current_data['connection_status'] = 'offline'

@app.websocket("/ws")
async def websocket_web(websocket: WebSocket):
    """WebSocket для веб-интерфейсов"""
    await websocket.accept()
    web_clients.add(websocket)
    print(f"Web client connected: {websocket.client}")
    
    try:
        # Отправляем текущие данные сразу
        await websocket.send_text(json.dumps(current_data))
        
        while True:
            # Поддерживаем соединение (ping/pong)
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        print(f"Web client disconnected: {websocket.client}")
        web_clients.remove(websocket)

async def broadcast_to_web_clients(data: Dict):
    """Рассылка данных всем веб-клиентам"""
    if web_clients:
        message = json.dumps(data)
        disconnected = set()
        
        for client in web_clients:
            try:
                await client.send_text(message)
            except:
                disconnected.add(client)
        
        # Удаляем отключенных клиентов
        web_clients.difference_update(disconnected)

@app.get("/health")
async def health_check():
    """Health check для мониторинга"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting AI Monitor Cloud API...")
    print("📡 WebSocket endpoints:")
    print("   /ws/laptop - for local laptop connection")
    print("   /ws - for web dashboard clients")
    print("🌐 Web interface: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
