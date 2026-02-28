# 🛡️ Tumar AI Monitor

Real-time AI-powered security monitoring system with distributed architecture.

## 🎯 Overview

Tumar AI Monitor is a comprehensive security solution that combines:
- **AI-powered object detection** using YOLOv8
- **Real-time risk assessment** with behavioral analysis
- **Distributed architecture** (laptop + cloud)
- **Modern React frontend** with WebSocket updates
- **Thermal integration** for enhanced detection

## 🏗️ Architecture

```
📹 Camera → 🖥️ Laptop (AI Processing) → 🌐 Cloud API → 🖥️ Frontend
```

### Components

1. **Laptop Client** (`laptop_client_local.py`)
   - Real-time video processing
   - AI object detection & tracking
   - Risk assessment
   - WebSocket data streaming

2. **Cloud API** (`render/api_integration_clean.py`)
   - FastAPI server with WebSocket support
   - REST API endpoints
   - Real-time data distribution
   - Static frontend serving

3. **Frontend** (`render/static/scene-insight-hub-91-main/`)
   - React + TypeScript + Vite
   - Real-time dashboard
   - WebSocket integration
   - Modern UI with Shadcn/ui

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Camera access
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tumar-ai-monitor.git
cd tumar-ai-monitor

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd render/static/scene-insight-hub-91-main
npm install
cd ../../..
```

### Running the System

1. **Start Cloud API Server**
```bash
cd render
python api_integration_clean.py
```
Server runs on: `http://localhost:8001`

2. **Start Frontend**
```bash
cd render/static/scene-insight-hub-91-main
npm run dev
```
Frontend runs on: `http://localhost:5173`

3. **Start Laptop Client**
```bash
python laptop_client_local.py
```

## 🌐 Cloud Deployment

### Render.com Deployment

1. **Push to GitHub**
2. **Create Render Web Service**
3. **Settings:**
   - Root Directory: `render`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python api_integration_clean.py`

### Environment Variables

- `PORT`: API server port (default: 8001)
- `CORS_ORIGINS`: Frontend URLs

## 📊 Features

### AI Capabilities
- **Object Detection**: YOLOv8n model
- **Multi-object Tracking**: ByteTrack + Kalman filter
- **Behavior Analysis**: Speed, direction, convoy detection
- **Risk Assessment**: Multi-factor scoring
- **Thermal Integration**: Weather-based thermal analysis

### Frontend Features
- **Real-time Video Stream**: AI-annotated video
- **Risk Gauge**: Visual risk indicator
- **Scene Statistics**: Object counts, speeds
- **Interactive Map**: Object positions
- **Alert System**: Risk-based notifications
- **Analytics Dashboard**: System performance metrics

### API Features
- **WebSocket**: Real-time bidirectional communication
- **REST Endpoints**: Data retrieval and system status
- **Health Checks**: System monitoring
- **CORS Support**: Cross-origin requests

## 🔧 Configuration

### Camera Settings
- Resolution: QQVGA (160x120)
- FPS: 10 (stable)
- Processing: 5 FPS AI

### Risk Thresholds
- **Normal**: 0.0 - 0.3
- **Monitor**: 0.3 - 0.6
- **Alert**: 0.6 - 1.0

### WebSocket Endpoints
- Laptop → Cloud: `/api/v1/ws/laptop`
- Frontend → Cloud: `/api/v1/ws/frontend`

## 📁 Project Structure

```
tumar-ai-monitor/
├── 🖥️ Laptop Client
│   ├── laptop_client_local.py    # Local testing
│   ├── laptop_client.py         # Cloud version
│   └── core/                     # AI modules
├── 🌐 Cloud Infrastructure
│   └── render/
│       ├── api_integration_clean.py    # API server
│       └── static/scene-insight-hub-91-main/  # Frontend
├── 📋 Documentation
│   ├── README.md
│   ├── START_GUIDE.md
│   └── FINAL_ARCHITECTURE.md
└── ⚙️ Configuration
    ├── requirements.txt
    ├── .gitignore
    └── package.json
```

## 🎮 Usage

1. **Start all components** following the Quick Start guide
2. **Open frontend** at `http://localhost:5173`
3. **Allow camera access** when prompted
4. **Monitor real-time data** in the dashboard
5. **View risk assessments** and alerts

## 🔍 Monitoring

### System Metrics
- FPS (Frames Per Second)
- Object detection counts
- Risk score trends
- System uptime

### Risk Factors
- Object speed anomalies
- Convoy detection
- Thermal activity
- Density analysis

## 🛠️ Development

### Adding New Features
1. Modify AI modules in `core/`
2. Update API endpoints in `render/api_integration_clean.py`
3. Extend frontend components in `render/static/scene-insight-hub-91-main/`

### Testing
```bash
# Run AI tests
python -m pytest tests/

# Run frontend tests
cd render/static/scene-insight-hub-91-main
npm test
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera not working**
   - Check Windows privacy settings
   - Ensure camera drivers are installed

2. **Port conflicts**
   ```bash
   netstat -ano | findstr :8001
   taskkill /PID <PID> /F
   ```

3. **Frontend not loading**
   - Run `npm install`
   - Check Node.js version

4. **No data in frontend**
   - Verify WebSocket connections
   - Check API server logs

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

🛡️ **Tumar AI Monitor** - Advanced Security Monitoring System
