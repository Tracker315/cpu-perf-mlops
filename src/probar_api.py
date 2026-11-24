"""Prueba la API sin levantar servidor (TestClient) y muestra salud, prediccion y metricas.

Correr:  python -m src.probar_api
"""
import json

from fastapi.testclient import TestClient

from .servir import app

with TestClient(app) as cliente:
    print("GET /health")
    print(json.dumps(cliente.get("/health").json(), indent=2))

    procesadores = [
        {"cpuName": "AMD Ryzen 9 5950X", "cores": 16, "TDP": 105, "testDate": 2021, "socket": "AM4", "category": "Desktop"},
        {"cpuName": "Intel Core i5-1135G7 @ 2.40GHz", "cores": 4, "TDP": 28, "testDate": 2021, "socket": "FCBGA1449", "category": "Laptop"},
        {"cpuName": "procesador inventado", "cores": 300, "TDP": None, "testDate": 2030, "category": "Server"},   # para ver alertas
    ]
    print("\nPOST /predict")
    print(json.dumps(cliente.post("/predict", json=procesadores).json(), indent=2, ensure_ascii=False))

    for _ in range(20):    # unas cuantas peticiones mas para tener latencias
        cliente.post("/predict", json=procesadores[:1])
    print("\nGET /metricas")
    print(json.dumps(cliente.get("/metricas").json(), indent=2))
