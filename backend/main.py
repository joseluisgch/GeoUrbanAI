"""
Geovisor IA Geográfico — Backend FastAPI
Autor: GEOPLANTER / Jose Luis Galindo Chillcce
Stack: FastAPI · osmnx · geopandas · rasterio · Google Gemini API (gemini-3.6-flash)
Sin dependencias compiladas: grilla hexagonal implementada con numpy + shapely.
"""

import json
import math
import os
import re
from io import BytesIO

import requests
import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from rasterio.transform import from_bounds
from scipy.stats import gaussian_kde
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Geovisor IA Geográfico",
    description="Análisis de accesibilidad urbana con OSM y Google Gemini AI (gemini-3.6-flash)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# ── Sistema de prompts del agente ──────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un agente geográfico especializado en análisis de accesibilidad urbana para ciudades peruanas y latinoamericanas.

Tu tarea es interpretar consultas en español y devolver SOLO un objeto JSON válido (sin texto adicional, sin markdown, sin explicaciones).

ESTRUCTURA REQUERIDA:
{
  "osm_tags": [{"key": "amenity", "value": "cafe"}],
  "analysis_type": "ELIGE_UNO_DE_LOS_4_TIPOS",
  "hex_size_m": 300,
  "buffer_m": 500,
  "interpretation": "Texto explicativo para el usuario en español",
  "layer_name": "Nombre corto para la leyenda"
}

REGLAS ESTRICTAS PARA analysis_type — elige EXACTAMENTE uno:

1. "density"
   → Cuando la consulta mencione: densidad, concentración, distribución, dónde hay más, cuántos hay, hexágonos.
   → Ejemplos: "zonas con más cafés", "dónde hay más restaurantes", "distribución de colegios".

2. "heatmap"
   → Cuando la consulta mencione EXPLÍCITAMENTE: mapa de calor, calor, heat map, intensidad, superficie, gradiente.
   → Ejemplos: "mapa de calor de restaurantes", "genera un heatmap", "muéstrame la intensidad de farmacias".
   → OBLIGATORIO usar "heatmap" si el usuario dice "mapa de calor" o "calor".

3. "comparison"
   → Cuando la consulta compare DOS tipos distintos de servicio.
   → Ejemplos: "restaurantes vs farmacias", "compara colegios con hospitales".

4. "isochrone"
   → Cuando la consulta mencione: accesible, cerca, radio, minutos a pie, cobertura, cuántos hay en X metros.
   → buffer_m: 300=4min, 500=7min, 1000=12min, 1500=18min.
   → Ejemplos: "hospitales a 500 metros", "qué hay cerca a pie".

HEX_SIZE_M (solo para density y comparison):
  150 = manzana, 300 = barrio (DEFAULT), 500 = zona amplia

TAGS OSM:
  Comida:     amenity=cafe, amenity=restaurant, amenity=fast_food, amenity=bar
  Comercio:   shop=supermarket, shop=bakery, amenity=marketplace
  Educación:  amenity=school, amenity=university, amenity=kindergarten
  Salud:      amenity=hospital, amenity=clinic, amenity=pharmacy
  Transporte: amenity=bus_station, highway=bus_stop, railway=station
  Recreación: leisure=park, leisure=playground, amenity=gym
  Servicios:  amenity=bank, amenity=atm, amenity=post_office

Responde SOLO con el JSON, nada más."""


# ── Modelos Pydantic ───────────────────────────────────────────────────────────
class BBox(BaseModel):
    north: float
    south: float
    east: float
    west: float


class QueryRequest(BaseModel):
    text: str
    bbox: BBox
    center: dict
    forced_type: str | None = None   # override desde el frontend
    include_zoning: bool | None = None
    zoning_district: str | None = None
    heat_radius_m: float | None = None
    heat_intensity_exponent: float | None = None
    heat_min_intensity: float | None = None
    heat_max_intensity: float | None = None


class RasterRequest(BaseModel):
    bbox: BBox
    osm_tags: list[dict]
    resolution: int = 256


# ── Grilla hexagonal (sin h3) ──────────────────────────────────────────────────
def _meters_to_deg_lat(meters: float) -> float:
    return meters / 111_000.0


def _meters_to_deg_lon(meters: float, lat: float) -> float:
    return meters / (111_000.0 * math.cos(math.radians(lat)))


def points_to_hex_grid(
    gdf: gpd.GeoDataFrame,
    bbox: "BBox",
    hex_size_m: float = 300,
) -> list[dict]:
    """
    Grilla hexagonal flat-top sobre el bbox, conteo por spatial join con geopandas.
    Sin h3 — solo numpy + shapely + geopandas (todos ya instalados).
    """
    if gdf.empty:
        return []

    mid_lat = (bbox.north + bbox.south) / 2.0
    ry = _meters_to_deg_lat(hex_size_m)
    rx = _meters_to_deg_lon(hex_size_m, mid_lat)
    dx = rx * 1.5
    dy = ry * math.sqrt(3)

    col_min = math.floor((bbox.west  - rx) / dx)
    col_max = math.ceil( (bbox.east  + rx) / dx)
    row_min = math.floor((bbox.south - ry) / dy)
    row_max = math.ceil( (bbox.north + ry) / dy)

    # Construir GeoDataFrame de hexágonos
    hex_records = []
    for col in range(col_min, col_max + 1):
        for row in range(row_min, row_max + 1):
            cx = col * dx
            cy = row * dy + (dy / 2 if col % 2 else 0)
            verts = [(cx + rx * math.cos(math.radians(60 * i)),
                      cy + ry * math.sin(math.radians(60 * i))) for i in range(6)]
            hex_records.append({"hex_id": f"{col}_{row}", "geometry": Polygon(verts)})

    if not hex_records:
        return []

    gdf_hex = gpd.GeoDataFrame(hex_records, crs="EPSG:4326")

    # Spatial join: asignar cada POI a su hexágono
    gdf_pts = gdf[["geometry"]].copy().reset_index(drop=True)
    joined = gpd.sjoin(gdf_pts, gdf_hex, how="left", predicate="within")
    counts = joined.groupby("hex_id").size().reset_index(name="count")

    if counts.empty:
        return []

    max_count = int(counts["count"].max())
    hex_by_id = {r["hex_id"]: r["geometry"] for r in hex_records}

    features = []
    for _, row in counts.iterrows():
        poly = hex_by_id.get(row["hex_id"])
        if poly is None:
            continue
        coords = list(poly.exterior.coords)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "count": int(row["count"]),
                "pct": round(int(row["count"]) / max_count * 100, 1),
                "hex_id": row["hex_id"],
            },
        })

    return features


# ── OSM ────────────────────────────────────────────────────────────────────────
def bbox_area_km2(bbox: BBox) -> float:
    lat_km = (bbox.north - bbox.south) * 111.0
    lon_km = (bbox.east - bbox.west) * 111.0 * np.cos(np.radians((bbox.north + bbox.south) / 2))
    return abs(lat_km * lon_km)


def fetch_osm_pois(bbox: BBox, osm_tags: list[dict]) -> gpd.GeoDataFrame:
    """Descarga POIs desde OSM usando osmnx 2.x (bbox orden W,S,E,N)."""
    tags_dict: dict = {}
    for tag in osm_tags:
        key, val = tag["key"], tag["value"]
        if key in tags_dict:
            existing = tags_dict[key]
            tags_dict[key] = existing + [val] if isinstance(existing, list) else [existing, val]
        else:
            tags_dict[key] = val

    try:
        gdf = ox.features_from_bbox(
            bbox=(bbox.west, bbox.south, bbox.east, bbox.north),
            tags=tags_dict,
        )
    except Exception as e:
        if any(m in str(e) for m in ["No data elements", "InsufficientResponseError", "no results"]):
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        raise

    gdf = gdf.copy()
    poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    gdf.loc[poly_mask, "geometry"] = gdf.loc[poly_mask, "geometry"].centroid
    gdf = gdf[gdf.geometry.geom_type == "Point"].to_crs("EPSG:4326")
    return gdf


ARCIS_ZONING_FEATURE_URL = (
    "https://services5.arcgis.com/bHvzrGGxW8wP6Utm/arcgis/rest/services/"
    "Zonficacion_Urbana_Vigente_LM_13022026_project/FeatureServer/0/query"
)


def _arcgis_rings_to_geojson_geometry(rings: list[list[list[float]]]) -> dict:
    if len(rings) == 1:
        return {"type": "Polygon", "coordinates": rings}
    return {"type": "MultiPolygon", "coordinates": [[ring] for ring in rings]}


def fetch_esri_zoning(bbox: BBox, district: str | None = None) -> list[dict]:
    """Consulta la capa de zonificación ESRI en el bbox y devuelve GeoJSON de polígonos."""
    where = "1=1"
    if district:
        safe = district.upper().replace("'", "''").strip()
        if safe:
            where = f"UPPER(nombdist) LIKE '%{safe}%'"
    params = {
        "where": where,
        "geometry": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "outFields": "simb_zoni,sclas_zoni,nombdist",
        "f": "json",
        "returnGeometry": "true",
        "spatialRel": "esriSpatialRelIntersects",
        "resultRecordCount": "2000",
    }

    resp = requests.get(ARCIS_ZONING_FEATURE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise ValueError(data["error"].get("message", "Error ArcGIS"))
    features = []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        if "rings" in geom:
            geometry = _arcgis_rings_to_geojson_geometry(geom["rings"])
        elif "paths" in geom:
            geometry = {"type": "MultiLineString", "coordinates": geom["paths"]}
        elif "x" in geom and "y" in geom:
            geometry = {"type": "Point", "coordinates": [geom["x"], geom["y"]]}
        else:
            continue

        attributes = feat.get("attributes", {})
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "simb_zoni": attributes.get("simb_zoni"),
                "sclas_zoni": attributes.get("sclas_zoni"),
                "nombdist": attributes.get("nombdist"),
            },
        })
    return features


POI_LABELS = {
    ("amenity", "cafe"): "Cafés",
    ("amenity", "restaurant"): "Restaurantes",
    ("amenity", "fast_food"): "Comidas rápidas",
    ("amenity", "bar"): "Bares",
    ("shop", "supermarket"): "Supermercados",
    ("shop", "bakery"): "Panaderías",
    ("amenity", "marketplace"): "Mercados",
    ("amenity", "school"): "Colegios",
    ("amenity", "university"): "Universidades",
    ("amenity", "kindergarten"): "Guarderías",
    ("amenity", "hospital"): "Hospitales",
    ("amenity", "clinic"): "Clínicas",
    ("amenity", "pharmacy"): "Farmacias",
    ("amenity", "bank"): "Bancos",
    ("amenity", "atm"): "Cajeros",
    ("leisure", "park"): "Parques",
    ("leisure", "playground"): "Plazas",
    ("amenity", "gym"): "Gimnasios",
}

def pretty_osm_tag(tag: dict) -> str:
    if not tag or "key" not in tag or "value" not in tag:
        return "POIs"
    return POI_LABELS.get((tag["key"], tag["value"]), f"{tag['value'].capitalize()}")


def gdf_to_kde_geotiff(gdf: gpd.GeoDataFrame, bbox: BBox, grid_size: int = 256) -> bytes:
    """Genera GeoTIFF KDE a partir de puntos."""
    lons = gdf.geometry.x.values
    lats = gdf.geometry.y.values

    x_grid = np.linspace(bbox.west, bbox.east, grid_size)
    y_grid = np.linspace(bbox.south, bbox.north, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])

    kernel = gaussian_kde(np.vstack([lons, lats]), bw_method=0.15)
    kde_values = kernel(positions).reshape(grid_size, grid_size)
    kde_norm = (kde_values / kde_values.max() * 255).astype(np.uint8)

    transform = from_bounds(bbox.west, bbox.south, bbox.east, bbox.north, grid_size, grid_size)
    buf = BytesIO()
    with rasterio.open(
        buf, "w", driver="GTiff",
        height=grid_size, width=grid_size,
        count=1, dtype=np.uint8,
        crs="EPSG:4326", transform=transform,
    ) as ds:
        ds.write(kde_norm[::-1], 1)
    return buf.getvalue()


def extract_minutes_from_text(text: str) -> list[int]:
    """
    Extrae los tiempos en minutos de una consulta en lenguaje natural.
    Soporta formatos como: '5 minutos', '5 y 10 minutos', '5, 10 o 15 minutos'.
    """
    text_lower = text.lower()
    found_times = set()
    
    # Buscar patrones de números seguidos de min/minutos/m
    for m in re.finditer(r'\b(\d+)\s*(?:minutos|min\b|mins\b)', text_lower):
        found_times.add(int(m.group(1)))
        # Buscar números anteriores en la misma frase (separados por comas, espacios, 'y', 'o')
        start_idx = m.start()
        lookback = text_lower[max(0, start_idx - 30):start_idx]
        for num in re.findall(r'\b(\d+)\b', lookback):
            found_times.add(int(num))
            
    # Filtrar solo tiempos lógicos para caminar (1 a 60 minutos)
    valid_times = [t for t in found_times if 1 <= t <= 60]
    return sorted(valid_times)


def compute_isochrone(
    center_lat: float,
    center_lon: float,
    buffer_m: float,
    gdf_pois: gpd.GeoDataFrame,
    use_manhattan: bool = False,
) -> dict:
    """
    Calcula la zona de accesibilidad a pie mediante buffer circular o Manhattan.
    """
    # 1. Determinar zona UTM a partir del centro
    utm_crs = _get_utm_crs(center_lat, center_lon)

    # 2. Crear GeoDataFrame del punto central y proyectar a UTM
    center_gdf = gpd.GeoDataFrame(
        {"geometry": [Point(center_lon, center_lat)]},
        crs="EPSG:4326",
    ).to_crs(utm_crs)

    # 3. Buffer en metros (círculo real o rombo de distancia Manhattan)
    pt = center_gdf.geometry.iloc[0]
    if use_manhattan:
        buffer_utm = Polygon([
            (pt.x + buffer_m, pt.y),
            (pt.x, pt.y + buffer_m),
            (pt.x - buffer_m, pt.y),
            (pt.x, pt.y - buffer_m),
            (pt.x + buffer_m, pt.y)
        ])
    else:
        buffer_utm = pt.buffer(buffer_m)

    # 4. Volver a WGS84 para exportar como GeoJSON
    buffer_gdf = gpd.GeoDataFrame({"geometry": [buffer_utm]}, crs=utm_crs).to_crs("EPSG:4326")
    zone_polygon = buffer_gdf.geometry.iloc[0]

    # 5. Filtrar POIs dentro del buffer (si hay POIs)
    pois_inside = []
    if not gdf_pois.empty:
        gdf_pois_utm = gdf_pois.to_crs(utm_crs)
        mask = gdf_pois_utm.geometry.within(buffer_utm)
        gdf_inside = gdf_pois[mask].copy()

        for _, row in gdf_inside.iterrows():
            name = row.get("name", "POI") if "name" in gdf_inside.columns else "POI"
            dist_m = center_gdf.geometry.iloc[0].distance(
                gdf_pois_utm.loc[row.name, "geometry"]
            )
            pois_inside.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row.geometry.x, row.geometry.y],
                },
                "properties": {
                    "name": str(name) if name and str(name) != "nan" else "POI",
                    "dist_m": round(dist_m, 0),
                    "walk_min": round(dist_m / 80, 1),  # ~80 m/min caminando
                },
            })

        # Ordenar por distancia
        pois_inside.sort(key=lambda f: f["properties"]["dist_m"])

    # 6. Construir GeoJSON del polígono de cobertura
    # Convertimos la geometría a GeoJSON nativo mediante __geo_interface__
    zone_feature = {
        "type": "Feature",
        "geometry": zone_polygon.__geo_interface__ if not zone_polygon.is_empty else None,
        "properties": {
            "buffer_m": buffer_m,
            "walk_min": round(buffer_m / 80, 1),
            "pois_inside": len(pois_inside),
            "isochrone_type": "manhattan" if use_manhattan else "circular",
            "is_center": True,
        },
    }

    return {
        "zone": zone_feature,
        "pois": pois_inside,
        "center": {"lat": center_lat, "lon": center_lon},
        "buffer_m": buffer_m,
        "total_inside": len(pois_inside),
    }


def compute_pois_isochrone(
    gdf_pois: gpd.GeoDataFrame,
    buffer_m: float,
    use_manhattan: bool = False,
) -> dict:
    """
    Calcula la zona de cobertura accesible alrededor de múltiples POIs.
    Une todos los buffers individuales en una sola área de cobertura.
    Retorna un diccionario estructurado similar a compute_isochrone.
    """
    if gdf_pois.empty:
        return {
            "features": [],
            "pois": [],
            "buffer_m": buffer_m,
            "total_inside": 0,
        }

    # 1. Determinar zona UTM usando el centro promedio de los POIs
    mean_lat = gdf_pois.geometry.y.mean()
    mean_lon = gdf_pois.geometry.x.mean()
    utm_crs = _get_utm_crs(mean_lat, mean_lon)

    # 2. Proyectar POIs a UTM
    gdf_pois_utm = gdf_pois.to_crs(utm_crs)

    # 3. Crear buffer para cada POI (circular o rombo Manhattan)
    polygons = []
    for _, row in gdf_pois_utm.iterrows():
        pt = row.geometry
        if use_manhattan:
            poly = Polygon([
                (pt.x + buffer_m, pt.y),
                (pt.x, pt.y + buffer_m),
                (pt.x - buffer_m, pt.y),
                (pt.x, pt.y - buffer_m),
                (pt.x + buffer_m, pt.y)
            ])
        else:
            poly = pt.buffer(buffer_m)
        polygons.append(poly)

    # 4. Unir todos los buffers
    union_poly = unary_union(polygons)

    # 5. Volver a WGS84
    union_gdf = gpd.GeoDataFrame({"geometry": [union_poly]}, crs=utm_crs).to_crs("EPSG:4326")
    zone_polygon = union_gdf.geometry.iloc[0]

    # Para evitar que el frontend falle con MultiPolygons en su cálculo de centroide,
    # descomponemos cualquier MultiPolygon resultante en Polígonos individuales.
    if zone_polygon.geom_type == "Polygon":
        polygons_list = [zone_polygon]
    elif zone_polygon.geom_type == "MultiPolygon":
        polygons_list = list(zone_polygon.geoms)
    else:
        polygons_list = []

    # Construir lista de features para la capa de cobertura
    zone_features = []
    for poly in polygons_list:
        if poly.is_empty:
            continue
        zone_features.append({
            "type": "Feature",
            "geometry": poly.__geo_interface__,
            "properties": {
                "buffer_m": buffer_m,
                "walk_min": round(buffer_m / 80, 1),
                "pois_inside": len(gdf_pois),
                "isochrone_type": "manhattan" if use_manhattan else "circular",
                "is_center": False,
            },
        })

    # Formatear POIs
    pois_features = []
    for _, row in gdf_pois.iterrows():
        name = row.get("name", "POI") if "name" in gdf_pois.columns else "POI"
        pois_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row.geometry.x, row.geometry.y],
            },
            "properties": {
                "name": str(name) if name and str(name) != "nan" else "POI",
                "dist_m": 0.0,
                "walk_min": 0.0,
            },
        })

    return {
        "features": zone_features,
        "pois": pois_features,
        "buffer_m": buffer_m,
        "total_inside": len(pois_features),
    }


def _get_utm_crs(lat: float, lon: float) -> str:
    """Retorna el CRS UTM más apropiado para las coordenadas dadas."""
    zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    # EPSG para UTM: 32600 + zona (norte) o 32700 + zona (sur)
    base = 32600 if hemisphere == "north" else 32700
    return f"EPSG:{base + zone}"


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "Geovisor IA Geográfico", "version": "1.1.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze(req: QueryRequest):
    """
    Endpoint principal: texto en lenguaje natural → GeoJSON enriquecido.
    Agente: Google Gemini gemini-3.6-flash.
    Hexágonos: grilla propia con numpy + shapely (sin h3).
    """
    area_km2 = bbox_area_km2(req.bbox)
    if area_km2 > 500:
        raise HTTPException(
            status_code=400,
            detail=f"Área demasiado grande ({area_km2:.0f} km²). Acerca el mapa.",
        )

    # 1. Agente Gemini
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no está configurada en el archivo .env del backend.")

    try:
        model = os.environ.get("GEMINI_MODEL", GEMINI_MODEL or "gemini-3.6-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        
        user_content = (
            f"Solicitud: '{req.text}'\n"
            f"Área visible: {area_km2:.1f} km²\n"
            f"Centro: lat={req.center.get('lat',0):.4f}, "
            f"lon={req.center.get('lon',0):.4f}"
        )
        
        payload = {
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json"
            }
        }
        
        # Intentar con retry en caso de 503 (alta demanda momentánea)
        max_attempts = 2
        res_json = None
        for attempt in range(max_attempts):
            res = requests.post(url, json=payload, timeout=50)
            if res.status_code == 503 and attempt < max_attempts - 1:
                import time
                time.sleep(2)
                continue
            res.raise_for_status()
            res_json = res.json()
            break
            
        try:
            ai_parts = res_json['candidates'][0]['content']['parts']
            ai_text = next((p['text'] for p in ai_parts if isinstance(p, dict) and 'text' in p), None)
            if not ai_text:
                raise ValueError("No text part found in response")
        except (KeyError, IndexError, ValueError, TypeError):
            raise HTTPException(status_code=500, detail="Respuesta vacía o formato incorrecto de Gemini API")
            
        intent = json.loads(ai_text.strip())

        # ── Override: prioridad 1 = frontend, prioridad 2 = palabras clave ──
        if req.forced_type:
            intent["analysis_type"] = req.forced_type
        else:
            text_lower = req.text.lower()
            heatmap_kw  = ["mapa de calor", "calor", "heatmap", "heat map", "intensidad", "gradiente", "superficie"]
            iso_kw      = ["a pie", "minutos", "radio", "metros de aquí", "cerca", "accesible", "cobertura"]
            compare_kw  = [" vs ", " versus ", "compara", "comparar", "diferencia entre"]
            zoning_kw   = ["zonificación", "uso del suelo", "simb_zoni", "sclas_zoni", "zonificacion"]
            if any(kw in text_lower for kw in heatmap_kw):
                intent["analysis_type"] = "heatmap"
            elif any(kw in text_lower for kw in compare_kw):
                intent["analysis_type"] = "comparison"
            elif any(kw in text_lower for kw in iso_kw):
                intent["analysis_type"] = "isochrone"
            if any(kw in text_lower for kw in zoning_kw):
                intent["include_zoning"] = True
    except json.JSONDecodeError as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"JSON inválido del agente: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error Gemini: {e}")

    # 2. Descargar POIs
    try:
        gdf = fetch_osm_pois(req.bbox, intent["osm_tags"])
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Error OSM: {e}")

    total_pois = len(gdf)
    analysis_type = intent.get("analysis_type", "density")
    include_zoning = bool(intent.get("include_zoning", req.include_zoning or False))
    hex_size_m = float(intent.get("hex_size_m", 300))
    buffer_m   = float(intent.get("buffer_m", 500))
    heat_radius_m = float(intent.get("heat_radius_m", req.heat_radius_m or 500.0))
    heat_intensity_exponent = float(intent.get("heat_intensity_exponent", req.heat_intensity_exponent or 2.0))
    heat_min_intensity = float(intent.get("heat_min_intensity", req.heat_min_intensity or 0.15))
    heat_max_intensity = float(intent.get("heat_max_intensity", req.heat_max_intensity or 1.0))

    # 3. Construir capas según tipo de análisis
    geojson_layers = []

    if analysis_type == "isochrone":
        # 1. Determinar métrica: Manhattan (calle/cuadrícula) o Euclidiana (circular)
        # Se activa Manhattan si se indica explícitamente "manhattan", "manhatan", "calles" o "isocrona" (por calles)
        text_lower = req.text.lower()
        use_manhattan = any(kw in text_lower for kw in ["manhattan", "manhatan", "calles", "isocrona por calles", "isócrona por calles"])

        # 2. Extraer todos los límites de tiempo solicitados (ej. 5, 10, 15 min)
        detected_times = extract_minutes_from_text(req.text)
        if not detected_times:
            # Fallback al buffer_m definido por el LLM o 500m
            buffer_m = float(intent.get("buffer_m", 500))
            times_buffers = [(round(buffer_m / 80), buffer_m)]
        else:
            # Mapeo exacto: 80 metros por minuto
            times_buffers = [(t, t * 80.0) for t in detected_times]

        # Ordenar buffers en orden descendente para que los polígonos más grandes se dibujen primero
        # y los más pequeños se dibujen encima, manteniéndose visibles
        times_buffers.sort(key=lambda x: x[1], reverse=True)

        pois_inside_list = []

        # 3. Para cada tiempo/distancia, calcular la cobertura accesible
        for mins, b_m in times_buffers:
            metric_label = "Manhattan" if use_manhattan else "Circular"
            
            if not gdf.empty:
                # Análisis basado en POIs: área de cobertura de todos los POIs
                iso_data = compute_pois_isochrone(gdf, b_m, use_manhattan=use_manhattan)
                features = iso_data["features"]
                pois_inside_list = iso_data["pois"]
                total_inside = iso_data["total_inside"]
            else:
                # Fallback: isocrona clásica desde el centro del visor
                center_lat = req.center.get("lat", (req.bbox.north + req.bbox.south) / 2)
                center_lon = req.center.get("lon", (req.bbox.east  + req.bbox.west)  / 2)
                iso_data = compute_isochrone(center_lat, center_lon, b_m, gdf, use_manhattan=use_manhattan)
                features = [iso_data["zone"]] if iso_data["zone"] else []
                pois_inside_list = iso_data["pois"]
                total_inside = iso_data["total_inside"]

            if features:
                geojson_layers.append({
                    "name": f"Cobertura accesible ({metric_label}) · {mins} min ({int(b_m)} m)",
                    "type": "isochrone",
                    "features": features,
                    "stats": {
                        "buffer_m": b_m,
                        "walk_min": float(mins),
                        "total_inside": total_inside,
                    },
                })

        # Capa 4: Mostrar los POIs analizados (una sola vez)
        if pois_inside_list:
            geojson_layers.append({
                "name": intent.get("layer_name", "POIs accesibles"),
                "type": "points_iso",
                "features": pois_inside_list[:2000],
                "stats": {"total": len(pois_inside_list)},
            })

    elif analysis_type == "heatmap" and total_pois > 0:
        # Heatmap: devolver puntos con peso para Leaflet.heat en el frontend
        # Intensidad normalizada por densidad local y ampliada para resaltar hotspots.
        lons = gdf.geometry.x.values
        lats = gdf.geometry.y.values

        heat_radius_deg = _meters_to_deg_lat(heat_radius_m)
        intensities = np.ones(len(gdf))
        if len(gdf) >= 10:
            from scipy.spatial import cKDTree
            coords = np.column_stack([lons, lats])
            tree = cKDTree(coords)
            neighbors = tree.query_ball_point(coords, r=heat_radius_deg)
            raw = np.array([len(n) for n in neighbors], dtype=float)
            if raw.max() > 0:
                normalized = raw / raw.max()
                # Aplicar exponente para enfatizar valores altos y reducir ruido bajo.
                scaled = np.power(normalized, heat_intensity_exponent)
                intensities = np.clip(scaled, heat_min_intensity, heat_max_intensity)

        heat_points = [
            [float(lat), float(lon), float(intensity)]
            for lat, lon, intensity in zip(lats, lons, intensities)
        ]
        geojson_layers.append({
            "name": intent.get("layer_name", "Mapa de calor"),
            "type": "heatmap",
            "heat_points": heat_points,   # [lat, lon, intensity]
            "heat_options": {
                "radius_m": heat_radius_m,
                "intensity_exponent": heat_intensity_exponent,
                "min_intensity": heat_min_intensity,
                "max_intensity": heat_max_intensity,
            },
            "features": [],               # vacío, no se usa GeoJSON aquí
            "stats": {
                "total": total_pois,
                "heat_radius_m": heat_radius_m,
                "heat_intensity_exponent": heat_intensity_exponent,
                "heat_min_intensity": heat_min_intensity,
                "heat_max_intensity": heat_max_intensity,
            },
        })

    elif analysis_type == "density" and total_pois > 0:
        hexagons = points_to_hex_grid(gdf, req.bbox, hex_size_m)
        if hexagons:
            base_name = intent.get("layer_name", "Densidad POIs")
            if not base_name.lower().startswith("hexágono") and not base_name.lower().startswith("hexagono"):
                layer_name = f"Hexágonos de {base_name}"
            else:
                layer_name = base_name
            geojson_layers.append({
                "name": layer_name,
                "type": "hexagons",
                "features": hexagons,
                "stats": {
                    "total": total_pois,
                    "max_per_hex": max(f["properties"]["count"] for f in hexagons),
                    "hex_count": len(hexagons),
                },
            })
        if total_pois <= 2000:
            points_layers = []
            for tag in intent.get("osm_tags", []):
                tag_key = tag.get("key")
                tag_value = tag.get("value")
                if not tag_key or tag_value is None:
                    continue
                if tag_key not in gdf.columns:
                    continue
                bucket = gdf[gdf[tag_key] == tag_value]
                if bucket.empty:
                    continue
                features = []
                for _, row in bucket.iterrows():
                    name = row.get("name", "POI") if "name" in bucket.columns else "POI"
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [row.geometry.x, row.geometry.y]},
                        "properties": {
                            "name": str(name) if name and str(name) != "nan" else "POI",
                            "poi_type": pretty_osm_tag(tag),
                        },
                    })
                points_layers.append({
                    "name": pretty_osm_tag(tag),
                    "type": "points",
                    "features": features,
                    "stats": {"total": len(features)},
                })

            if len(points_layers) > 1:
                geojson_layers.extend(points_layers)
            elif points_layers:
                geojson_layers.append(points_layers[0])
            else:
                points = []
                for _, row in gdf.iterrows():
                    name = row.get("name", "POI") if "name" in gdf.columns else "POI"
                    points.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [row.geometry.x, row.geometry.y]},
                        "properties": {"name": str(name) if name and str(name) != "nan" else "POI"},
                    })
                geojson_layers.append({"name": "Puntos individuales", "type": "points", "features": points, "stats": {"total": len(points)}})

    if include_zoning:
        try:
            zoning_district = intent.get("zoning_district", req.zoning_district)
            zoning_features = fetch_esri_zoning(req.bbox, zoning_district)
            if zoning_features:
                zoning_layer = {
                    "name": "Zonificación de usos del suelo",
                    "type": "zoning",
                    "features": zoning_features,
                    "stats": {"total": len(zoning_features), "district": zoning_district},
                }
                # Si el usuario solicitó zonificación explícitamente, devolver SOLO la capa de zonificación
                geojson_layers = [zoning_layer]
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error ArcGIS zonificación: {e}")

    return {
        "success": True,
        "interpretation": intent.get("interpretation", "Análisis completado."),
        "analysis_type": analysis_type,
        "total_pois": total_pois,
        "area_km2": round(area_km2, 2),
        "hex_size_m": hex_size_m,
        "buffer_m": buffer_m,
        "layers": geojson_layers,
        "osm_tags_used": intent.get("osm_tags", []),
        "warnings": (
            ["Más de 2000 POIs. Solo se muestran hexágonos agregados."]
            if total_pois > 2000 else []
        ),
    }


class IsochroneRequest(BaseModel):
    center_lat: float
    center_lon: float
    buffer_m: float = 500
    osm_tags: list[dict]
    use_manhattan: bool = False


@app.post("/isochrone")
async def isochrone(req: IsochroneRequest):
    """
    Endpoint dedicado para isocronas.
    Recibe centro + radio + tags OSM, devuelve zona + POIs dentro.
    """
    # Bbox ampliado: descargar POIs en un área generosa alrededor del centro
    pad = _meters_to_deg_lat(req.buffer_m * 1.5)
    pad_lon = _meters_to_deg_lon(req.buffer_m * 1.5, req.center_lat)
    bbox = BBox(
        north=req.center_lat + pad,
        south=req.center_lat - pad,
        east=req.center_lon + pad_lon,
        west=req.center_lon - pad_lon,
    )
    try:
        gdf = fetch_osm_pois(bbox, req.osm_tags)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error OSM: {e}")

    iso = compute_isochrone(req.center_lat, req.center_lon, req.buffer_m, gdf, use_manhattan=req.use_manhattan)
    return {
        "success": True,
        "center": iso["center"],
        "buffer_m": iso["buffer_m"],
        "total_inside": iso["total_inside"],
        "zone": iso["zone"],
        "pois": iso["pois"][:500],   # top 500 más cercanos
    }


@app.post("/raster/kde")
async def raster_kde(req: RasterRequest):
    gdf = fetch_osm_pois(req.bbox, req.osm_tags)
    if gdf.empty:
        raise HTTPException(status_code=404, detail="Sin datos en el área.")
    if len(gdf) < 5:
        raise HTTPException(status_code=422, detail="Muy pocos puntos para KDE.")
    return Response(content=gdf_to_kde_geotiff(gdf, req.bbox, req.resolution), media_type="image/tiff")


@app.get("/tags/search")
def search_tags(q: str = ""):
    catalog = [
        {"key": "amenity", "value": "cafe",         "label": "Cafeterías"},
        {"key": "amenity", "value": "restaurant",    "label": "Restaurantes"},
        {"key": "amenity", "value": "fast_food",     "label": "Comida rápida"},
        {"key": "amenity", "value": "bar",           "label": "Bares"},
        {"key": "amenity", "value": "hospital",      "label": "Hospitales"},
        {"key": "amenity", "value": "clinic",        "label": "Clínicas"},
        {"key": "amenity", "value": "pharmacy",      "label": "Farmacias"},
        {"key": "amenity", "value": "school",        "label": "Colegios"},
        {"key": "amenity", "value": "university",    "label": "Universidades"},
        {"key": "amenity", "value": "bank",          "label": "Bancos"},
        {"key": "amenity", "value": "atm",           "label": "Cajeros ATM"},
        {"key": "amenity", "value": "bus_station",   "label": "Terminales de bus"},
        {"key": "amenity", "value": "marketplace",   "label": "Mercados"},
        {"key": "shop",    "value": "supermarket",   "label": "Supermercados"},
        {"key": "shop",    "value": "bakery",        "label": "Panaderías"},
        {"key": "leisure", "value": "park",          "label": "Parques"},
        {"key": "leisure", "value": "playground",    "label": "Juegos infantiles"},
        {"key": "landuse", "value": "residential",   "label": "Zona residencial"},
        {"key": "landuse", "value": "commercial",    "label": "Zona comercial"},
        {"key": "landuse", "value": "industrial",    "label": "Zona industrial"},
    ]
    if not q:
        return catalog
    q_lower = q.lower()
    return [t for t in catalog if q_lower in t["label"].lower()
            or q_lower in t["value"].lower() or q_lower in t["key"].lower()]
