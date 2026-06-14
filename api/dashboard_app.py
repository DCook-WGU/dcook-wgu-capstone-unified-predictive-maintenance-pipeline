from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
PROJECT_ROOT=Path(os.environ.get('CAPSTONE_PROJECT_ROOT','/workspace')).resolve(); ARTIFACT_ROOT=PROJECT_ROOT/'artifacts/gold/pump'
app=FastAPI(title='Capstone Pump Anomaly Dashboard', version='0.1.0')
def read_json(path: Path) -> dict[str,Any]:
    if not path.exists(): raise HTTPException(status_code=404, detail=f'Missing artifact: {path}')
    return json.loads(path.read_text(encoding='utf-8'))
def read_csv_records(path: Path, limit:int=100) -> list[dict[str,Any]]:
    if not path.exists(): raise HTTPException(status_code=404, detail=f'Missing artifact: {path}')
    return pd.read_csv(path).head(limit).to_dict(orient='records')
@app.get('/health')
def health(): return {'status':'ok','project_root':str(PROJECT_ROOT)}
@app.get('/metrics/comparison')
def comparison_metrics(): return read_csv_records(ARTIFACT_ROOT/'comparison/results/pump__gold__model_comparison.csv', limit=50)
@app.get('/metrics/anomaly-detection')
def anomaly_detection_summary(): return read_json(ARTIFACT_ROOT/'anomaly_detection/summaries/stage3_improved__detection_summary.json')
@app.get('/packets/top-alerts')
def top_alert_packets(limit:int=25): return read_csv_records(ARTIFACT_ROOT/'anomaly_detection/packets/stage3_improved__top_alert_packets.csv', limit=limit)
@app.get('/', response_class=HTMLResponse)
def dashboard() -> str:
    comp=ARTIFACT_ROOT/'comparison/results/pump__gold__model_comparison.csv'; summ=ARTIFACT_ROOT/'anomaly_detection/summaries/stage3_improved__detection_summary.json'
    comp_html=pd.read_csv(comp).to_html(index=False) if comp.exists() else '<p>Comparison CSV not found.</p>'
    summ_html='<pre>'+json.dumps(json.loads(summ.read_text()),indent=2)+'</pre>' if summ.exists() else '<p>Gold 05 summary JSON not found.</p>'
    return f'<html><head><title>Capstone Pump Anomaly Dashboard</title><style>body{{font-family:Arial,sans-serif;margin:2rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:.4rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}pre{{background:#f6f6f6;padding:1rem;overflow-x:auto}}</style></head><body><h1>Capstone Pump Anomaly Dashboard</h1><p>Lightweight artifact dashboard. Reads existing Gold 04 and Gold 05 outputs; does not train models.</p><h2>Gold 04 Model Comparison</h2>{comp_html}<h2>Gold 05 Early-Warning Summary</h2>{summ_html}</body></html>'
