# Geovisor IA Geográfico

Geovisor de accesibilidad urbana con agente de inteligencia artificial geográfico.
El usuario escribe en lenguaje natural y el sistema analiza densidades, distribución
y accesibilidad de servicios usando datos de **OpenStreetMap** + **Claude AI**.

```
Arquitectura
  frontend/index.html   → GitHub Pages (gratuito)
  backend/main.py       → FastAPI en Render.com (gratuito)
```

---

## Requisitos previos

- Cuenta en [GitHub](https://github.com)
- Cuenta en [Render.com](https://render.com)
- Cuenta en [xAI Console](https://console.x.ai) (API key de Grok)
- Python 3.11+ (para desarrollo local)

---

## Despliegue local (desarrollo)

### 1. Backend

```bash
cd backend

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar API key de xAI
export XAI_API_KEY="xai-..."

# Levantar servidor
uvicorn main:app --reload --port 8000
```

Verifica en: http://localhost:8000/docs (Swagger UI automático)

### 2. Frontend

Abre `frontend/index.html` en tu navegador directamente, o sirve con:

```bash
cd frontend
python -m http.server 3000
```

Ingresa `http://localhost:8000` en el campo de URL del panel y haz clic en **Conectar**.

---

## Despliegue en producción (gratuito)

### Backend → Render.com

1. Sube el repositorio a GitHub
2. En [render.com](https://render.com) → **New → Web Service**
3. Conecta tu repositorio
4. Configuración:
   - **Root Directory**: `backend`
   - **Runtime**: Docker
   - **Plan**: Free
5. Variables de entorno:
   - `XAI_API_KEY` = tu clave de xAI (console.x.ai)
6. Haz clic en **Deploy**

> ⚠️ El plan gratuito de Render "duerme" el servicio tras 15 min de inactividad.
> La primera consulta puede tardar ~30 segundos en despertar la instancia.
> Para evitarlo, usa [UptimeRobot](https://uptimerobot.com) con un ping cada 10 min.

### Frontend → GitHub Pages

1. Ve a **Settings → Pages** en tu repositorio
2. Source: **Deploy from a branch**
3. Branch: `main`, folder: `/frontend`
4. Tu geovisor estará en: `https://tu-usuario.github.io/geovisor-ia/`

Recuerda actualizar la URL del backend en el campo de configuración del panel.

---

## Estructura del proyecto

```
geovisor-ia/
├── backend/
│   ├── main.py           # FastAPI: endpoints + agente IA + análisis geoespacial
│   ├── requirements.txt  # Dependencias Python
│   └── Dockerfile        # Contenedor para Render
├── frontend/
│   └── index.html        # Interfaz completa (mapa + chat)
├── render.yaml           # Configuración de despliegue Render
└── README.md
```

---

## Endpoints de la API

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Info del servicio |
| GET | `/health` | Estado del servidor |
| POST | `/analyze` | **Principal**: analiza consulta en lenguaje natural |
| POST | `/raster/kde` | Genera GeoTIFF KDE para heatmaps continuos |
| GET | `/tags/search?q=...` | Busca tags OSM por texto |

### Ejemplo de llamada a `/analyze`

```json
POST /analyze
{
  "text": "¿Dónde hay más cafés y restaurantes?",
  "bbox": {
    "north": -12.040,
    "south": -12.055,
    "east": -77.030,
    "west": -77.055
  },
  "center": { "lat": -12.046, "lon": -77.042 }
}
```

### Respuesta

```json
{
  "success": true,
  "interpretation": "Se analizó la distribución de cafeterías y restaurantes...",
  "analysis_type": "density",
  "total_pois": 87,
  "area_km2": 3.42,
  "h3_resolution": 8,
  "layers": [
    {
      "name": "Cafés y restaurantes",
      "type": "hexagons",
      "features": [...],
      "stats": { "total": 87, "max_per_hex": 12, "hex_count": 23 }
    }
  ]
}
```

---

## Tipos de análisis disponibles

| Tipo | Descripción |
|------|-------------|
| `density` | Conteo de POIs por hexágono H3 con escala de color |
| `heatmap` | Mapa de calor KDE (requiere endpoint `/raster/kde`) |
| `comparison` | Comparación de dos grupos de POIs |
| `isochrone` | Zona accesible en radio determinado |

---

## Tags OSM utilizables

El agente conoce los tags más comunes de OSM para Perú/Latinoamérica:

- **Alimentación**: `amenity=cafe`, `amenity=restaurant`, `amenity=fast_food`
- **Salud**: `amenity=hospital`, `amenity=clinic`, `amenity=pharmacy`
- **Educación**: `amenity=school`, `amenity=university`
- **Transporte**: `amenity=bus_station`, `highway=bus_stop`
- **Comercio**: `shop=supermarket`, `amenity=marketplace`
- **Recreación**: `leisure=park`, `leisure=playground`
- **Servicios**: `amenity=bank`, `amenity=atm`

---

## Extensiones posibles (Fase 2+)

- [ ] Análisis de isocrona a pie con `osmnx.isochrone`
- [ ] Índice de diversidad de usos (Shannon entropy por hexágono)
- [ ] Integración con datos del INEI (shapefiles de distritos)
- [ ] Exportación a GeoPackage / Shapefile con `geopandas`
- [ ] Análisis de autocorrelación espacial (Moran's I) con `esda`
- [ ] Persistencia de consultas con Supabase PostGIS
- [ ] Dashboard de métricas agregadas por distrito

---

## Créditos

- **Datos**: [OpenStreetMap](https://www.openstreetmap.org) contributors
- **IA**: [xAI Grok](https://x.ai/api)
- **Mapa**: [Leaflet.js](https://leafletjs.com) + [CARTO](https://carto.com)
- **Índices H3**: [Uber H3](https://h3geo.org)
- **Análisis geoespacial**: [osmnx](https://osmnx.readthedocs.io), [geopandas](https://geopandas.org), [rasterio](https://rasterio.readthedocs.io)
