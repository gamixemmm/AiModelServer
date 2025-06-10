@echo off
echo Starting SAP AI Assistant...
set PORT=8000
uvicorn api:app --host 0.0.0.0 --port %PORT% 