"""
test_api.py — Script de prueba rápida del backend
Ejecutar con: python test_api.py

Requiere backend corriendo en localhost:8000
"""

import json
import requests

BASE_URL = "http://localhost:8000"

# ── Bbox de Miraflores, Lima ───────────────────────────────────────────────────
BBOX_MIRAFLORES = {
    "north": -12.110,
    "south": -12.130,
    "east": -77.018,
    "west": -77.040,
}

CENTER_MIRAFLORES = {"lat": -12.120, "lon": -77.029}

QUERIES = [
    "¿En qué zonas hay más cafés y restaurantes?",
    "Muéstrame la densidad de farmacias y clínicas",
    "¿Dónde hay parques y zonas verdes?",
    "Distribución de colegios en el área",
]


def test_health():
    print("🔍 Verificando /health ...")
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"Health check falló: {r.status_code}"
    print("  ✅ OK:", r.json())


def test_tags():
    print("\n🔍 Probando /tags/search ...")
    r = requests.get(f"{BASE_URL}/tags/search?q=cafe")
    assert r.status_code == 200
    tags = r.json()
    print(f"  ✅ {len(tags)} tags encontrados para 'cafe'")
    for t in tags[:3]:
        print(f"     {t['key']}={t['value']} → {t['label']}")


def test_analyze(query: str):
    print(f"\n🧠 Analizando: '{query}'")
    payload = {
        "text": query,
        "bbox": BBOX_MIRAFLORES,
        "center": CENTER_MIRAFLORES,
    }
    r = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=60)
    if r.status_code != 200:
        print(f"  ❌ Error {r.status_code}: {r.text[:200]}")
        return

    data = r.json()
    hex_info = f"Hex-{data['hex_size_m']}m" if 'hex_size_m' in data else "GIS"
    print(f"  ✅ {data['total_pois']} POIs · {data['area_km2']} km² · {hex_info}")
    print(f"  📝 {data['interpretation'][:120]}...")

    for layer in data.get("layers", []):
        n_feat = len(layer.get("features", []))
        print(f"  🗂  Capa '{layer['name']}' ({layer['type']}): {n_feat} features")
        if layer.get("stats"):
            print(f"     Stats: {layer['stats']}")

    if data.get("warnings"):
        for w in data["warnings"]:
            print(f"  ⚠️  {w}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Geovisor IA — Test de API")
    print("=" * 55)

    test_health()
    test_tags()

    for q in QUERIES[:2]:   # Solo 2 para no consumir mucha API
        test_analyze(q)

    print("\n" + "=" * 55)
    print("  Pruebas completadas")
    print("=" * 55)
