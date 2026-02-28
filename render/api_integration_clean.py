from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
import json
import asyncio
from typing import Dict, List
import time
from datetime import datetime

app = FastAPI(title="SafeQunar API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    'connection_status': 'offline',
    'system_info': {
        'uptime': 0,
        'total_frames': 0,
        'alerts_count': 0
    }
}

web_clients: List[WebSocket] = []
laptop_client: WebSocket = None
risk_history = []
system_start_time = time.time()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=/static/">
    <title>SafeQunar - Redirecting...</title>
</head>
<body>
    <h1>Redirecting to SafeQunar Dashboard...</h1>
    <p>If not redirected, <a href="/static/">click here</a></p>
</body>
</html>
    """)

@app.get("/api/v1/status")
async def get_status():
    uptime = time.time() - system_start_time
    
    return JSONResponse({
        "status": "online",
        "service": "SafeQunar AI Monitor",
        "version": "1.0.0",
        "timestamp": current_data['timestamp'],
        "connection_status": current_data['connection_status'],
        "web_clients": len(web_clients),
        "laptop_connected": laptop_client is not None,
        "uptime": uptime,
        "data_age": time.time() - current_data['timestamp'] if current_data['timestamp'] > 0 else None,
        "system_info": current_data['system_info']
    })

@app.get("/api/v1/current")
async def get_current_data():
    return JSONResponse(current_data)

@app.get("/api/v1/risk-history")
async def get_risk_history(limit: int = 50):
    return JSONResponse({
        "history": risk_history[-limit:],
        "total_count": len(risk_history),
        "current_risk": current_data['risk_score'],
        "current_level": current_data['risk_level']
    })

@app.get("/api/v1/analytics")
async def get_analytics():
    uptime = time.time() - system_start_time
    
    if risk_history:
        avg_risk = sum(item['risk_score'] for item in risk_history) / len(risk_history)
        max_risk = max(item['risk_score'] for item in risk_history)
        risk_levels = {
            'normal': len([r for r in risk_history if r['risk_level'] == 'NORMAL']),
            'monitor': len([r for r in risk_history if r['risk_level'] == 'MONITOR']),
            'alert': len([r for r in risk_history if r['risk_level'] == 'ALERT'])
        }
    else:
        avg_risk = max_risk = 0
        risk_levels = {'normal': 0, 'monitor': 0, 'alert': 0}
    
    return JSONResponse({
        "uptime": uptime,
        "total_frames": current_data['system_info']['total_frames'],
        "alerts_count": current_data['system_info']['alerts_count'],
        "risk_stats": {
            "average": avg_risk,
            "maximum": max_risk,
            "levels": risk_levels
        },
        "performance": {
            "current_fps": current_data['fps'],
            "data_rate": "300 KB/s",
            "connection_quality": "good" if laptop_client else "offline"
        },
        "scene": current_data['scene']
    })

@app.post("/api/v1/data")
async def receive_data(data: Dict):
    global current_data, risk_history
    
    current_data.update(data)
    current_data['timestamp'] = time.time()
    current_data['connection_status'] = 'online'
    
    current_data['system_info']['uptime'] = time.time() - system_start_time
    current_data['system_info']['total_frames'] = current_data.get('frame_id', 0)
    
    if 'risk_score' in data:
        risk_history.append({
            'timestamp': current_data['timestamp'],
            'risk_score': data['risk_score'],
            'risk_level': data.get('risk_level', 'NORMAL')
        })
        
        if data.get('risk_level') == 'ALERT':
            current_data['system_info']['alerts_count'] += 1
        
        if len(risk_history) > 1000:
            risk_history.pop(0)
    
    await broadcast_to_web_clients(data)
    
    return JSONResponse({
        "status": "received", 
        "timestamp": current_data['timestamp'],
        "processed": True
    })

@app.get("/api/v1/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "service": "SafeQunar API",
        "timestamp": time.time(),
        "version": "1.0.0",
        "dependencies": {
            "websocket": "operational",
            "data_processing": "operational",
            "laptop_connection": "connected" if laptop_client else "disconnected"
        }
    })

@app.websocket("/api/v1/ws/laptop")
async def websocket_laptop(websocket: WebSocket):
    global laptop_client
    
    await websocket.accept()
    laptop_client = websocket
    print(f"Laptop connected: {websocket.client}")
    
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
                
                if len(risk_history) > 1000:
                    risk_history.pop(0)
            
            await broadcast_to_web_clients(message)
            
    except WebSocketDisconnect:
        print(f"Laptop disconnected: {websocket.client}")
        laptop_client = None
        current_data['connection_status'] = 'offline'

@app.websocket("/api/v1/ws/frontend")
async def websocket_frontend(websocket: WebSocket):
    await websocket.accept()
    web_clients.add(websocket)
    print(f"Frontend connected: {websocket.client}")
    
    try:
        await websocket.send_text(json.dumps({
            "type": "initial_data",
            "data": current_data
        }))
        
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        print(f"Frontend disconnected: {websocket.client}")
        web_clients.remove(websocket)

async def broadcast_to_web_clients(data: Dict):
    if web_clients:
        message = json.dumps({
            "type": "data_update",
            "data": data,
            "timestamp": time.time()
        })
        
        disconnected = set()
        for client in web_clients:
            try:
                await client.send_text(message)
            except:
                disconnected.add(client)
        
        web_clients.difference_update(disconnected)

@app.get("/static/")
async def static_index():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>SafeQunar Frontend Integration</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 40px; 
            background: #f5f5f5; 
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }
        .api-info { 
            background: #e3f2fd; 
            padding: 20px; 
            border-radius: 5px; 
            margin: 20px 0; 
        }
        .endpoint { 
            background: #f3e5f5; 
            padding: 10px; 
            margin: 10px 0; 
            border-radius: 3px; 
            font-family: monospace; 
        }
        .status { 
            padding: 10px; 
            border-radius: 5px; 
            margin: 10px 0; 
        }
        .online { background: #c8e6c9; color: #2e7d32; }
        .offline { background: #ffcdd2; color: #c62828; }
    </style>
</head>
<body>
    <div class="container">
        <h1>SafeQunar Frontend Integration</h1>
        
        <div class="api-info">
            <h3>API Endpoints</h3>
            <div class="endpoint">GET /api/v1/status - System status</div>
            <div class="endpoint">GET /api/v1/current - Current data</div>
            <div class="endpoint">GET /api/v1/risk-history - Risk history</div>
            <div class="endpoint">GET /api/v1/analytics - Analytics</div>
            <div class="endpoint">POST /api/v1/data - Receive laptop data</div>
            <div class="endpoint">WS /api/v1/ws/laptop - Laptop connection</div>
            <div class="endpoint">WS /api/v1/ws/frontend - Frontend connection</div>
        </div>
        
        <h3>Integration Steps</h3>
        <ol>
            <li>Use frontend_clean.js for integration</li>
            <li>Use REST API endpoints for data fetching</li>
            <li>Connect to WebSocket for real-time updates</li>
            <li>Replace fake data with real API calls</li>
        </ol>
        
        <h3>Current Status</h3>
        <div id="status" class="status">Loading...</div>
        
        <h3>WebSocket Example</h3>
        <pre><code>
const ws = new WebSocket('wss://SafeQunar.onrender.com/api/v1/ws/frontend');
ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'data_update') {
        updateDashboard(message.data);
    }
};
        </code></pre>
    </div>
    
    <script>
        fetch('/api/v1/status')
            .then(response => response.json())
            .then(data => {
                const statusDiv = document.getElementById('status');
                if (data.laptop_connected) {
                    statusDiv.className = 'status online';
                    statusDiv.textContent = 'Laptop Connected - System Operational';
                } else {
                    statusDiv.className = 'status offline';
                    statusDiv.textContent = 'Laptop Offline - Waiting for Connection';
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    </script>
</body>
</html>
    """)

if __name__ == "__main__":
    import uvicorn
    
    print("SafeQunar API Integration Server")
    print("Frontend Integration Ready")
    print("API Endpoints: /api/v1/*")
    print("WebSocket: /api/v1/ws/*")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
