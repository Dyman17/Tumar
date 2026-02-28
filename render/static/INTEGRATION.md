# SafeQunar Frontend Integration

## Структура проекта

```
scene-insight-hub-91-main/
├── src/
│   ├── hooks/
│   │   └── useSafeQunarAPI.ts    # API интеграция
│   ├── pages/
│   │   ├── SafeQunarIndex.tsx     # Основная страница с API
│   │   └── Index.tsx             # Mock страница
│   └── App.tsx                   # Обновлен для маршрутизации
├── package.json                  # Зависимости
└── vite.config.ts               # Конфигурация
```

## Интеграция с SafeQunar API

### 1. Новый hook: `useSafeQunarAPI`

```typescript
import { useSafeQunarAPI } from "@/hooks/useSafeQunarAPI";

function MyComponent() {
  const { data, objects, alerts, connected, analytics } = useSafeQunarAPI();
  
  // Данные в реальном времени:
  // - data.riskScore - уровень риска
  // - data.vehicles - количество машин
  // - data.persons - количество людей
  // - data.convoyFlag - флаг конвоя
  // - data.thermalFlag - флаг тепловых аномалий
  // - connected - статус подключения
  // - analytics - системная аналитика
}
```

### 2. API Endpoints

- **WebSocket**: `wss://SafeQunar.onrender.com/api/v1/ws/frontend`
- **REST**: `https://SafeQunar.onrender.com/api/v1/*`
  - `/current` - текущие данные
  - `/analytics` - аналитика
  - `/risk-history` - история риска
  - `/status` - статус системы

### 3. Маршрутизация

- `/` - SafeQunarIndex (реальные данные)
- `/mock` - Index (mock данные для разработки)

## Установка и запуск

```bash
# В папке scene-insight-hub-91-main
npm install
npm run dev
```

## Особенности

### Real-time данные
- Автопереподключение WebSocket
- Сглаживание риска
- Алерты при высоком риске
- История событий

### Аналитика
- Uptime системы
- Общее количество кадров
- Количество алертов
- Средний риск
- FPS

### Адаптивный дизайн
- Mobile-friendly
- Tailwind CSS
- Shadcn/ui компоненты

## Деплои

Фронтенд готов к деплою на любой платформе:
- Vercel
- Netlify
- GitHub Pages
- Render

## Конфигурация

Для изменения API URL:
```typescript
// В useSafeQunarAPI.ts
const API_BASE = 'https://your-domain.com/api/v1';
const WS_URL = 'wss://your-domain.com/api/v1/ws/frontend';
```

## Компоненты

### SafeQunarIndex
- Основной дашборд
- Real-time видео
- Risk gauge
- Статистика сцены
- Карта объектов
- Алерты
- Аналитика

### useSafeQunarAPI hook
- WebSocket подключение
- REST API запросы
- Автопереподключение
- Управление состоянием
