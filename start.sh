#!/bin/bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
sleep 8
streamlit run app/frontend.py --server.port 7860 --server.address 0.0.0.0 --server.headless true