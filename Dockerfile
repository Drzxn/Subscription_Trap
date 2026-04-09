# 🔹 Base image
FROM python:3.10-slim

# 🔹 Prevent buffering (good for logs)
ENV PYTHONUNBUFFERED=1

# 🔹 Set working directory
WORKDIR /app

# 🔹 Copy requirements first (for caching)
COPY requirements.txt .

# 🔹 Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 🔹 Copy rest of project
COPY . .

# 🔹 Expose HF required port
EXPOSE 7860

# 🔹 Start FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]