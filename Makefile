.PHONY: setup data fetch train test eval api web demo
setup:  ; cd backend && pip install -r requirements.txt && cd ../frontend && npm install
data:   ; cd backend && python scripts/generate_data.py --train 600
fetch:  ; cd backend && python scripts/fetch_public_data.py
train:  ; cd backend && python scripts/train_public_models.py
test:   ; cd backend && AERONEXUS_STATE_DB=:memory: pytest -q && cd ../frontend && npm test && npm run typecheck
eval:   ; cd backend && AERONEXUS_STATE_DB=:memory: python -m aeronexus.evals.runner
api:    ; cd backend && uvicorn aeronexus.api:app --reload --port 8000
web:    ; cd frontend && npm run dev
demo:   ; docker compose up --build
