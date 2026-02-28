import { useState, useEffect, useCallback } from 'react';

interface SceneData {
  riskScore: number;
  vehicles: number;
  persons: number;
  convoyFlag: boolean;
  thermalFlag: boolean;
  avgSpeed?: number;
  fps?: number;
}

interface MapObject {
  id: string;
  type: 'vehicle' | 'person';
  x: number;
  y: number;
  angle: number;
  risk: number;
}

interface Alert {
  id: number;
  time: string;
  message: string;
  severity: 'low' | 'medium' | 'high';
}

const API_BASE = 'https://SafeQunar.onrender.com/api/v1';
const WS_URL = 'wss://SafeQunar.onrender.com/api/v1/ws/frontend';

class SafeQunarAPI {
  private ws: WebSocket | null = null;
  private listeners: Map<string, Function[]> = new Map();

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.ws = new WebSocket(WS_URL);

    this.ws.onopen = () => {
      console.log('Connected to SafeQunar API');
      this.emit('connected', true);
    };

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        if (message.type === 'data_update') {
          this.emit('data', message.data);
        } else if (message.type === 'initial_data') {
          this.emit('data', message.data);
        }
      } catch (error) {
        console.error('WebSocket message error:', error);
      }
    };

    this.ws.onclose = () => {
      console.log('Disconnected from SafeQunar API');
      this.emit('connected', false);
      
      setTimeout(() => this.connect(), 3000);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.emit('connected', false);
    };
  }

  on(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(callback);
  }

  off(event: string, callback: Function) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  private emit(event: string, data: any) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }

  async getCurrentData() {
    try {
      const response = await fetch(`${API_BASE}/current`);
      if (!response.ok) throw new Error('API error');
      return await response.json();
    } catch (error) {
      console.error('Error fetching current data:', error);
      return null;
    }
  }

  async getAnalytics() {
    try {
      const response = await fetch(`${API_BASE}/analytics`);
      if (!response.ok) throw new Error('API error');
      return await response.json();
    } catch (error) {
      console.error('Error fetching analytics:', error);
      return null;
    }
  }

  async getRiskHistory(limit = 50) {
    try {
      const response = await fetch(`${API_BASE}/risk-history?limit=${limit}`);
      if (!response.ok) throw new Error('API error');
      return await response.json();
    } catch (error) {
      console.error('Error fetching risk history:', error);
      return null;
    }
  }
}

const safeQunarAPI = new SafeQunarAPI();

export { SafeQunarAPI };
export { safeQunarAPI as api };
export function useSafeQunarAPI() {
  const [data, setData] = useState<SceneData>({
    riskScore: 0,
    vehicles: 0,
    persons: 0,
    convoyFlag: false,
    thermalFlag: false,
    avgSpeed: 0,
    fps: 0
  });
  
  const [objects, setObjects] = useState<MapObject[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [connected, setConnected] = useState(false);
  const [analytics, setAnalytics] = useState<any>(null);

  const handleData = useCallback((apiData: any) => {
    const sceneData: SceneData = {
      riskScore: apiData.risk_score || 0,
      vehicles: apiData.scene?.vehicles || 0,
      persons: apiData.scene?.persons || 0,
      convoyFlag: apiData.scene?.convoy_detected || false,
      thermalFlag: apiData.scene?.thermal_active || false,
      avgSpeed: apiData.scene?.avg_speed || 0,
      fps: apiData.fps || 0
    };

    setData(sceneData);

    if (apiData.video_frame) {
      setObjects([]);
    }

    if (sceneData.riskScore > 0.7) {
      const newAlert: Alert = {
        id: Date.now(),
        time: new Date().toLocaleTimeString(),
        message: sceneData.convoyFlag 
          ? `Convoy detected: ${sceneData.vehicles} vehicles`
          : `High risk: ${sceneData.riskScore.toFixed(2)} — ${sceneData.persons} person(s)`,
        severity: sceneData.riskScore > 0.85 ? 'high' : 'medium'
      };
      
      setAlerts(prev => [...prev.slice(-5), newAlert]);
    }
  }, []);

  useEffect(() => {
    safeQunarAPI.connect();
    
    safeQunarAPI.on('connected', setConnected);
    safeQunarAPI.on('data', handleData);

    return () => {
      safeQunarAPI.off('connected', setConnected);
      safeQunarAPI.off('data', handleData);
    };
  }, [handleData]);

  useEffect(() => {
    const loadInitialData = async () => {
      const currentData = await safeQunarAPI.getCurrentData();
      if (currentData) {
        handleData(currentData);
      }

      const analyticsData = await safeQunarAPI.getAnalytics();
      if (analyticsData) {
        setAnalytics(analyticsData);
      }
    };

    loadInitialData();
  }, [handleData]);

  return {
    data,
    objects,
    alerts,
    connected,
    analytics,
    isLive: true
  };
}
