#!/usr/bin/env bash
#
# start.sh -- launch the CertifAI backend and frontend together (macOS / Linux).
#
# Both servers run in this shell; Ctrl-C stops both.
#
# Usage:  ./start.sh

set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python_bin="$root/backend/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "No virtual environment at backend/.venv. Create one with:" >&2
  echo "  cd backend && python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# The API refuses to predict without trained artifacts, so warn before starting
# rather than letting the frontend surface a 503.
if [[ ! -f "$root/backend/models/certifai_models.joblib" ]]; then
  echo "Warning: no trained model at backend/models/certifai_models.joblib." >&2
  echo "Run 'python train_model.py' in backend/ first (about 3 minutes)." >&2
fi

if [[ ! -d "$root/frontend/node_modules" ]]; then
  echo "Frontend dependencies missing. Run 'npm install' in frontend/." >&2
  exit 1
fi

# Make sure neither server outlives this script.
trap 'kill 0' EXIT INT TERM

echo "Starting backend on http://127.0.0.1:8000 ..."
(cd "$root/backend" && "$python_bin" -m uvicorn app:app --port 8000 --reload) &

echo "Starting frontend on http://localhost:5173 ..."
(cd "$root/frontend" && npm run dev) &

echo
echo "EMC Advisor is starting. Open http://localhost:5173"
wait
