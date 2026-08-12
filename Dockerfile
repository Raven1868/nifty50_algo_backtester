# Reproducible environment for the NIFTY 50 backtester. Pinning the base
# image and installing exact requirements.txt versions here means "works on
# my machine" surprises (e.g. the yfinance MultiIndex-columns behavior that
# differed across yfinance versions) are caught inside a controlled image
# rather than whatever pip happens to resolve on a fresh local install.
FROM python:3.12-slim

WORKDIR /app

# System deps for matplotlib (headless rendering, already Agg-backend in
# code) and openpyxl's XML handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persist cache/output directories as volumes so results survive container
# restarts — mount these from the host, e.g.:
#   docker run -v $(pwd)/data/raw:/app/data/raw -v $(pwd)/reports:/app/reports ...
VOLUME ["/app/data/raw", "/app/reports", "/app/charts", "/app/logs"]

# Default: run a backtest. Override the command for other entry points, e.g.:
#   docker run <image> streamlit run visualization/dashboard.py --server.address=0.0.0.0
#   docker run <image> python -m pytest tests/ -v
ENTRYPOINT ["python", "main.py"]
CMD ["--strategy", "momentum", "--universe", "nifty50"]
