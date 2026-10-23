FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd --create-home --uid 1000 app \
    && chown -R app:app /app

USER app

EXPOSE 8503

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.1", "--server.port=8503", "--server.headless=true", "--browser.gatherUsageStats=false"]