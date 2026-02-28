# 🛡️ SafeQunar AI Monitor

**Real-time AI Intelligence & Risk Assessment Platform**

## 🌐 Deployment

This folder contains the cloud API for deployment on Render.com at:
**https://SafeQunar.onrender.com/**

## 📁 Files Structure

```
render/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
└── README.md         # This file
```

## 🚀 Quick Deploy to Render

### 1. Push to GitHub
```bash
git add render/
git commit -m "Add SafeQunar cloud API"
git push origin main
```

### 2. Create Render Web Service
1. Go to [Render.com](https://render.com)
2. New → Web Service
3. Connect your GitHub repository
4. Configure:
   - **Name**: `SafeQunar`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Root Directory**: `render`
   - **HTTP Port**: 8000

### 3. Deploy
- Click "Create Web Service"
- Wait for deployment (2-3 minutes)
- Access at: `https://SafeQunar.onrender.com`

## 🔧 Configuration

### Environment Variables (Optional)
```
PORT=8000
HOST=0.0.0.0
```

### Custom Domain
In Render dashboard:
1. Go to Custom Domains
2. Add `SafeQunar.onrender.com`
3. Configure DNS if needed

## 📡 API Endpoints

### Web Interface
```
GET  /                    # Main dashboard
GET  /health             # Health check
GET  /api/status         # System status
```

### WebSocket
```
/ws/laptop              # Laptop client connection
/ws                     # Web dashboard clients
```

## 📱 Laptop Client Setup

Update the URL in `laptop_client.py`:
```python
cloud_url = "wss://SafeQunar.onrender.com/ws/laptop"
```

## 🛡️ Features

### Real-time Dashboard
- Live video stream with AI overlay
- Risk assessment matrix
- Advanced analytics
- Responsive design
- Auto-reconnection

### Security Features
- WebSocket encryption (WSS)
- CORS protection
- Health monitoring
- Rate limiting ready

### Performance
- Optimized for Render free tier
- Minimal resource usage
- Fast WebSocket connections
- Efficient data streaming

## 🔍 Monitoring

### Health Check
```bash
curl https://SafeQunar.onrender.com/health
```

### System Status
```bash
curl https://SafeQunar.onrender.com/api/status
```

## 🚨 Troubleshooting

### Common Issues
1. **WebSocket connection failed**
   - Check URL in laptop client
   - Verify Render service is running
   - Check firewall/proxy settings

2. **Deployment failed**
   - Verify requirements.txt format
   - Check Python version compatibility
   - Review Render build logs

3. **Slow performance**
   - Optimize video quality settings
   - Reduce FPS on laptop client
   - Check Render resource limits

### Logs
- Render Dashboard → Logs tab
- Real-time error tracking
- Connection status monitoring

## 💰 Pricing

### Render Free Tier
- 750 hours/month (enough for 24/7)
- 512MB RAM limit
- Shared CPU
- No credit card required

### Resource Usage
- Memory: ~50-100MB
- CPU: < 5% average
- Bandwidth: ~300 KB/s per client

## 🔒 Security

### Production Recommendations
1. Add API key authentication
2. Implement rate limiting
3. Use HTTPS (automatic on Render)
4. Monitor for abuse
5. Regular security updates

### Authentication Example
```python
# Add to app.py
API_KEYS = ["your-secret-key"]

@app.websocket("/ws/laptop")
async def websocket_laptop(websocket: WebSocket):
    token = await websocket.receive_text()
    if token not in API_KEYS:
        await websocket.close(code=4001)
        return
```

## 📈 Analytics

### Metrics Tracked
- Connection uptime
- Risk score trends
- Object detection counts
- System performance
- Data transfer rates

### Export Data
```javascript
// In browser console
localStorage.setItem('riskHistory', JSON.stringify(riskHistory));
```

## 🎯 Next Steps

1. **Deploy to Render** using the steps above
2. **Test laptop connection** with updated URL
3. **Monitor performance** in Render dashboard
4. **Scale resources** if needed
5. **Add custom domain** for branding

## 📞 Support

For issues:
1. Check Render logs
2. Verify laptop client configuration
3. Test WebSocket connection
4. Monitor system resources

---

**SafeQunar AI Monitor** - Next-Generation Security Intelligence 🛡️
