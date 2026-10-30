# AgroIA-COVIO v2.1 — Streamlit sobre Hugging Face Spaces (SDK: docker, CPU)
FROM python:3.10-slim

# Dependencias de sistema: git (para instalar mobile-sam) + libs de OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces ejecuta el contenedor como el usuario 1000.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME="/home/user/app/.cache" \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Instalar dependencias primero (mejor cache de capas).
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copiar el resto de la app.
COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0"]
