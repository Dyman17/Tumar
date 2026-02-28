class SafeQunarAPI {
    constructor(baseUrl = 'https://SafeQunar.onrender.com') {
        this.baseUrl = baseUrl;
        this.ws = null;
        this.callbacks = {};
    }

    async getStatus() {
        const response = await fetch(`${this.baseUrl}/api/v1/status`);
        return await response.json();
    }

    async getCurrentData() {
        const response = await fetch(`${this.baseUrl}/api/v1/current`);
        return await response.json();
    }

    async getRiskHistory(limit = 50) {
        const response = await fetch(`${this.baseUrl}/api/v1/risk-history?limit=${limit}`);
        return await response.json();
    }

    async getAnalytics() {
        const response = await fetch(`${this.baseUrl}/api/v1/analytics`);
        return await response.json();
    }

    connectWebSocket() {
        const wsUrl = `${this.baseUrl.replace('http', 'ws')}/api/v1/ws/frontend`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('Connected to SafeQunar API');
            this.onConnectionChange?.(true);
        };

        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            
            switch (message.type) {
                case 'initial_data':
                    this.onInitialData?.(message.data);
                    break;
                case 'data_update':
                    this.onDataUpdate?.(message.data);
                    break;
                default:
                    console.log('Unknown message type:', message.type);
            }
        };

        this.ws.onclose = () => {
            console.log('Disconnected from SafeQunar API');
            this.onConnectionChange?.(false);
            
            setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    onConnectionChange(connected) {
        console.log('Connection status:', connected ? 'Connected' : 'Disconnected');
    }

    onInitialData(data) {
        console.log('Initial data received:', data);
        this.updateDashboard(data);
    }

    onDataUpdate(data) {
        console.log('Data update received:', data);
        this.updateDashboard(data);
    }

    updateDashboard(data) {
        this.updateRiskDisplay(data.risk_score, data.risk_level);
        this.updateSceneStats(data.scene);
        this.updateVideoFrame(data.video_frame);
        this.updateSystemInfo(data);
    }

    updateRiskDisplay(score, level) {
        const riskElement = document.getElementById('risk-score');
        const riskBar = document.getElementById('risk-bar');
        
        if (riskElement) {
            riskElement.textContent = `${score.toFixed(2)} - ${level}`;
            riskElement.className = `risk-score ${level.toLowerCase()}`;
        }
        
        if (riskBar) {
            riskBar.style.width = `${score * 100}%`;
        }
    }

    updateSceneStats(scene) {
        const elements = {
            vehicles: document.getElementById('vehicles-count'),
            persons: document.getElementById('persons-count'),
            speed: document.getElementById('avg-speed'),
            convoy: document.getElementById('convoy-status'),
            thermal: document.getElementById('thermal-status')
        };

        if (elements.vehicles) elements.vehicles.textContent = scene.vehicles || 0;
        if (elements.persons) elements.persons.textContent = scene.persons || 0;
        if (elements.speed) elements.speed.textContent = (scene.avg_speed || 0).toFixed(1);
        if (elements.convoy) elements.convoy.textContent = scene.convoy_detected ? 'YES' : 'NO';
        if (elements.thermal) elements.thermal.textContent = scene.thermal_active ? 'YES' : 'NO';
    }

    updateVideoFrame(frameData) {
        const videoElement = document.getElementById('video-stream');
        if (videoElement && frameData) {
            videoElement.src = `data:image/jpeg;base64,${frameData}`;
        }
    }

    updateSystemInfo(data) {
        const fpsElement = document.getElementById('fps-display');
        const statusElement = document.getElementById('connection-status');
        
        if (fpsElement) fpsElement.textContent = (data.fps || 0).toFixed(1);
        if (statusElement) statusElement.textContent = 'Online';
    }

    async loadHistoricalData() {
        try {
            const history = await this.getRiskHistory(100);
            const analytics = await this.getAnalytics();
            
            this.drawRiskChart(history.history);
            this.updateAnalyticsPanel(analytics);
        } catch (error) {
            console.error('Error loading historical data:', error);
        }
    }

    drawRiskChart(history) {
        const canvas = document.getElementById('risk-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        ctx.clearRect(0, 0, width, height);

        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth = 2;
        ctx.beginPath();

        history.forEach((point, index) => {
            const x = (index / history.length) * width;
            const y = height - (point.risk_score * height);
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();
    }

    updateAnalyticsPanel(analytics) {
        const elements = {
            uptime: document.getElementById('uptime'),
            totalFrames: document.getElementById('total-frames'),
            alertsCount: document.getElementById('alerts-count'),
            avgRisk: document.getElementById('avg-risk')
        };

        if (elements.uptime) {
            const hours = Math.floor(analytics.uptime / 3600);
            const minutes = Math.floor((analytics.uptime % 3600) / 60);
            elements.uptime.textContent = `${hours}h ${minutes}m`;
        }
        
        if (elements.totalFrames) elements.totalFrames.textContent = analytics.total_frames;
        if (elements.alertsCount) elements.alertsCount.textContent = analytics.alerts_count;
        if (elements.avgRisk) elements.avgRisk.textContent = analytics.risk_stats.average.toFixed(3);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.safeQunarAPI = new SafeQunarAPI();
    
    window.safeQunarAPI.connectWebSocket();
    
    setTimeout(() => {
        window.safeQunarAPI.loadHistoricalData();
    }, 1000);
    
    setInterval(async () => {
        try {
            const status = await window.safeQunarAPI.getStatus();
            console.log('System status:', status);
        } catch (error) {
            console.error('Error fetching status:', error);
        }
    }, 10000);
});

if (typeof module !== 'undefined' && module.exports) {
    module.exports = SafeQunarAPI;
}
