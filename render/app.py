"""
SafeQunar.onrender.com - Cloud API for AI Monitor
Optimized for Render deployment
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from typing import Dict, List
import time
from datetime import datetime

app = FastAPI(title="SafeQunar AI Monitor", version="1.0.0")

# CORS для подключения ноутбука
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
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

web_clients: List[WebSocket] = []
laptop_client: WebSocket = None
risk_history = []
max_history = 50

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Главная страница - SafeQunar Dashboard"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SafeQunar - AI Monitor</title>
    <meta name="description" content="Real-time AI monitoring and risk assessment system">
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🛡️</text></svg>">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #ffffff;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .stars {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            background: transparent url('data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="100" height="100"%3E%3Ccircle cx="50" cy="50" r="1" fill="white" opacity="0.3"/%3E%3C/svg%3E') repeat;
            animation: drift 200s linear infinite;
        }
        
        @keyframes drift {
            from { transform: translate(0, 0); }
            to { transform: translate(-100px, -100px); }
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            position: relative;
            z-index: 1;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .logo {
            font-size: 3em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #00d4ff, #090979, #00d4ff);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient 3s ease infinite;
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .tagline {
            font-size: 1.2em;
            opacity: 0.8;
            margin-bottom: 20px;
        }
        
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255,255,255,0.05);
            padding: 20px 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 500;
        }
        
        .status-dot {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .status-online { 
            background: #00ff88; 
            box-shadow: 0 0 20px #00ff88;
        }
        
        .status-offline { 
            background: #ff4757; 
            box-shadow: 0 0 20px #ff4757;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        @media (max-width: 968px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }
        
        .card h3 {
            margin-bottom: 20px;
            font-size: 1.4em;
            color: #00d4ff;
        }
        
        .video-container {
            text-align: center;
        }
        
        .video-stream {
            width: 100%;
            max-width: 500px;
            border-radius: 15px;
            border: 2px solid rgba(0,212,255,0.3);
            box-shadow: 0 10px 30px rgba(0,212,255,0.2);
        }
        
        .risk-container {
            position: relative;
        }
        
        .risk-bar {
            width: 100%;
            height: 40px;
            background: rgba(0,0,0,0.3);
            border-radius: 20px;
            overflow: hidden;
            margin: 20px 0;
            position: relative;
        }
        
        .risk-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff88, #ffd700, #ff4757);
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 20px;
            position: relative;
            overflow: hidden;
        }
        
        .risk-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        .risk-score {
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
            text-shadow: 0 0 20px currentColor;
        }
        
        .risk-normal { color: #00ff88; }
        .risk-monitor { color: #ffd700; }
        .risk-alert { color: #ff4757; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 20px;
        }
        
        .stat-item {
            text-align: center;
            padding: 20px;
            background: rgba(255,255,255,0.03);
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.05);
            transition: all 0.3s ease;
        }
        
        .stat-item:hover {
            background: rgba(255,255,255,0.08);
            transform: translateY(-2px);
        }
        
        .stat-value {
            font-size: 2.2em;
            font-weight: bold;
            color: #00d4ff;
            margin-bottom: 8px;
            text-shadow: 0 0 10px currentColor;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .alert-banner {
            background: linear-gradient(135deg, rgba(255,71,87,0.2), rgba(255,71,87,0.1));
            border: 1px solid #ff4757;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            display: none;
            animation: alertPulse 1s infinite;
        }
        
        .alert-banner.show {
            display: block;
        }
        
        @keyframes alertPulse {
            0%, 100% { 
                background: linear-gradient(135deg, rgba(255,71,87,0.2), rgba(255,71,87,0.1));
                border-color: #ff4757;
            }
            50% { 
                background: linear-gradient(135deg, rgba(255,71,87,0.4), rgba(255,71,87,0.2));
                border-color: #ff6b7a;
            }
        }
        
        .loading {
            text-align: center;
            padding: 60px;
            font-size: 1.2em;
            opacity: 0.7;
        }
        
        .loading::after {
            content: '';
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .footer {
            text-align: center;
            margin-top: 50px;
            padding: 30px;
            opacity: 0.6;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        
        .brand {
            font-size: 1.1em;
            font-weight: 600;
            color: #00d4ff;
        }
    </style>
</head>
<body>
    <div class="stars"></div>
    
    <div class="container">
        <div class="header">
            <div class="logo">🛡️ SafeQunar</div>
            <div class="tagline">Real-time AI Intelligence & Risk Assessment</div>
            <div style="font-size: 0.9em; opacity: 0.6;">
                Powered by Advanced Computer Vision & Machine Learning
            </div>
        </div>
        
        <div class="status-bar">
            <div class="status-indicator">
                <div id="statusDot" class="status-dot status-offline"></div>
                <span id="statusText">Initializing...</span>
            </div>
            <div>
                <span id="timestamp">System starting...</span>
            </div>
        </div>
        
        <div id="alertBanner" class="alert-banner">
            <strong>⚠️ SECURITY ALERT:</strong> 
            <span id="alertMessage">Elevated risk levels detected</span>
        </div>
        
        <div class="main-grid">
            <div class="card video-container">
                <h3>📹 Live Intelligence Feed</h3>
                <div id="loadingVideo" class="loading">Establishing secure connection...</div>
                <img id="videoStream" class="video-stream" style="display:none;" src="" alt="AI video stream">
                <div style="margin-top: 15px; font-size: 0.9em; opacity: 0.7;">
                    <span>🔥 FPS: </span><span id="fpsDisplay">0.0</span>
                    <span style="margin-left: 20px;">📊 Frame: </span><span id="frameId">0</span>
                </div>
            </div>
            
            <div class="card risk-container">
                <h3>🚨 Risk Assessment Matrix</h3>
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
                        <div class="stat-label">Speed</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="convoyStatus">NO</div>
                        <div class="stat-label">Convoy</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>📊 Advanced Analytics</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" id="thermalStatus">NO</div>
                    <div class="stat-label">Thermal</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="uptime">0s</div>
                    <div class="stat-label">Uptime</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="dataRate">0 KB/s</div>
                    <div class="stat-label">Data Rate</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="riskTrend">→</div>
                    <div class="stat-label">Trend</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <div class="brand">SafeQunar AI Monitor</div>
            <div style="margin-top: 10px; font-size: 0.9em;">
                Next-Generation Security Intelligence Platform
            </div>
        </div>
    </div>

    <script>
        let ws;
        let reconnectInterval;
        let startTime = Date.now();
        let lastDataTime = 0;
        let dataBytes = 0;
        let riskHistory = [];
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('✅ Connected to SafeQunar Cloud');
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
                
                // Сохраняем историю риска
                if (data.risk_score) {
                    riskHistory.push(data.risk_score);
                    if (riskHistory.length > 10) riskHistory.shift();
                    updateRiskTrend();
                }
            };
            
            ws.onclose = function() {
                console.log('❌ Disconnected from SafeQunar Cloud');
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
            document.getElementById('fpsDisplay').textContent = (data.fps || 0).toFixed(1);
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
                convoyElement.style.color = '#ff4757';
                convoyElement.style.textShadow = '0 0 10px #ff4757';
            } else {
                convoyElement.style.color = '#00d4ff';
                convoyElement.style.textShadow = '0 0 10px #00d4ff';
            }
            
            const thermalElement = document.getElementById('thermalStatus');
            if (scene.thermal_active) {
                thermalElement.style.color = '#ff4757';
                thermalElement.style.textShadow = '0 0 10px #ff4757';
            } else {
                thermalElement.style.color = '#00d4ff';
                thermalElement.style.textShadow = '0 0 10px #00d4ff';
            }
        }
        
        function updateRiskTrend() {
            const trendElement = document.getElementById('riskTrend');
            if (riskHistory.length < 2) {
                trendElement.textContent = '→';
                return;
            }
            
            const recent = riskHistory.slice(-3);
            const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
            const prev = riskHistory.slice(-6, -3);
            if (prev.length === 0) {
                trendElement.textContent = '→';
                return;
            }
            
            const prevAvg = prev.reduce((a, b) => a + b, 0) / prev.length;
            
            if (avg > prevAvg + 0.1) {
                trendElement.textContent = '↑';
                trendElement.style.color = '#ff4757';
            } else if (avg < prevAvg - 0.1) {
                trendElement.textContent = '↓';
                trendElement.style.color = '#00ff88';
            } else {
                trendElement.textContent = '→';
                trendElement.style.color = '#00d4ff';
            }
        }
        
        function updateConnectionStatus(connected) {
            const indicator = document.getElementById('statusDot');
            const text = document.getElementById('statusText');
            
            if (connected) {
                indicator.className = 'status-dot status-online';
                text.textContent = '🟢 Online';
            } else {
                indicator.className = 'status-dot status-offline';
                text.textContent = '🔴 Offline';
            }
        }
        
        function updateTimestamp() {
            const now = new Date();
            document.getElementById('timestamp').textContent = now.toLocaleTimeString();
            
            // Uptime
            const uptime = Math.floor((Date.now() - startTime) / 1000);
            const hours = Math.floor(uptime / 3600);
            const minutes = Math.floor((uptime % 3600) / 60);
            const seconds = uptime % 60;
            
            if (hours > 0) {
                document.getElementById('uptime').textContent = `${hours}h ${minutes}m`;
            } else if (minutes > 0) {
                document.getElementById('uptime').textContent = `${minutes}m ${seconds}s`;
            } else {
                document.getElementById('uptime').textContent = `${seconds}s`;
            }
            
            // Data rate
            if (lastDataTime > 0) {
                const timeDiff = (Date.now() - lastDataTime) / 1000;
                const dataRate = timeDiff > 0 ? (dataBytes / 1024 / timeDiff).toFixed(1) : '0';
                document.getElementById('dataRate').textContent = dataRate + ' KB/s';
            }
        }
        
        // Запуск
        window.onload = function() {
            connectWebSocket();
            setInterval(updateTimestamp, 1000);
        };
        
        // Очистка
        window.onbeforeunload = function() {
            if (ws) ws.close();
            if (reconnectInterval) clearInterval(reconnectInterval);
        };
    </script>
</body>
</html>
    """

@app.post("/api/data")
async def receive_data(data: Dict):
    """Прием данных от ноутбука"""
    global current_data, risk_history
    
    current_data.update(data)
    current_data['timestamp'] = time.time()
    current_data['connection_status'] = 'online'
    
    if 'risk_score' in data:
        risk_history.append({
            'timestamp': current_data['timestamp'],
            'risk_score': data['risk_score'],
            'risk_level': data.get('risk_level', 'NORMAL')
        })
        
        if len(risk_history) > max_history:
            risk_history.pop(0)
    
    await broadcast_to_web_clients(data)
    
    return {"status": "received", "timestamp": current_data['timestamp']}

@app.get("/api/status")
async def get_status():
    """Статус системы"""
    return {
        "status": "online",
        "service": "SafeQunar AI Monitor",
        "version": "1.0.0",
        "timestamp": current_data['timestamp'],
        "connection_status": current_data['connection_status'],
        "web_clients": len(web_clients),
        "laptop_connected": laptop_client is not None,
        "data_age": time.time() - current_data['timestamp'] if current_data['timestamp'] > 0 else None
    }

@app.get("/health")
async def health_check():
    """Health check для Render"""
    return {
        "status": "healthy",
        "service": "SafeQunar",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@app.websocket("/ws/laptop")
async def websocket_laptop(websocket: WebSocket):
    """Подключение ноутбука"""
    global laptop_client
    
    await websocket.accept()
    laptop_client = websocket
    print(f"📱 Laptop connected: {websocket.client}")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            current_data.update(message)
            current_data['timestamp'] = time.time()
            current_data['connection_status'] = 'online'
            
            if 'risk_score' in message:
                risk_history.append({
                    'timestamp': current_data['timestamp'],
                    'risk_score': message['risk_score'],
                    'risk_level': message.get('risk_level', 'NORMAL')
                })
                
                if len(risk_history) > max_history:
                    risk_history.pop(0)
            
            await broadcast_to_web_clients(message)
            
    except WebSocketDisconnect:
        print(f"📱 Laptop disconnected: {websocket.client}")
        laptop_client = None
        current_data['connection_status'] = 'offline'

@app.websocket("/ws")
async def websocket_web(websocket: WebSocket):
    """Подключение веб-клиентов"""
    await websocket.accept()
    web_clients.add(websocket)
    print(f"🌐 Web client connected: {websocket.client}")
    
    try:
        await websocket.send_text(json.dumps(current_data))
        
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        print(f"🌐 Web client disconnected: {websocket.client}")
        web_clients.remove(websocket)

async def broadcast_to_web_clients(data: Dict):
    """Рассылка данных веб-клиентам"""
    if web_clients:
        message = json.dumps(data)
        disconnected = set()
        
        for client in web_clients:
            try:
                await client.send_text(message)
            except:
                disconnected.add(client)
        
        web_clients.difference_update(disconnected)

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 SafeQunar AI Monitor - Cloud API")
    print("🌐 URL: https://SafeQunar.onrender.com")
    print("📡 WebSocket endpoints:")
    print("   /ws/laptop - for laptop connection")
    print("   /ws - for web dashboard")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
