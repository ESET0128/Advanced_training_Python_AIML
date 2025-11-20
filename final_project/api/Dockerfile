FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# (Optional but recommended) system deps for scikit-learn builds
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# If you have requirements.txt in api/, prefer this:
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip \
#     && pip install --no-cache-dir -r requirements.txt

# Or keep this simple version:
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn[standard] pandas scikit-learn joblib python-multipart jinja2

# Copy all API code into the image
COPY . .

# FastAPI will listen on 9000 inside the container
EXPOSE 9000

# IMPORTANT: main.py must be in /app and contain `app = FastAPI()`
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
