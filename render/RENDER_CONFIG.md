# 🚀 Render Configuration Update

## 📋 Изменения в Build Command:

### **Текущая Build Command:**
```
pip install -r requirements.txt
```

### **Новая Build Command:**
```
pip install -r requirements.txt
cd static/scene-insight-hub-91-main && npm install && npm run build && cd ../..
```

## 🔧 Что делает новая команда:

1. **Устанавливает Python зависимости**
2. **Переходит в папку фронтенда**
3. **Устанавливает Node.js зависимости** (`npm install`)
4. **Собирает фронтенд** (`npm run build`)
5. **Возвращается в корень**

## 📁 Результат:

- Фронтенд будет собран в `static/scene-insight-hub-91-main/dist/`
- API будет обслуживать файлы из этой папки
- Сайт будет доступен по `https://tumar.onrender.com/static/`

## ⚡ Альтернативный вариант:

Если первая команда не сработает, попробуй:

```
pip install -r requirements.txt && cd static/scene-insight-hub-91-main && npm install && npm run build
```

Или раздели на две команды:

**Build Command:**
```
pip install -r requirements.txt
```

**Pre-Deploy Command:**
```
cd static/scene-insight-hub-91-main && npm install && npm run build
```
