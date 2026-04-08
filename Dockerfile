# Use lightweight Python
FROM python:3.10-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy entire project
COPY . .

# Install dependencies via pyproject (IMPORTANT)
RUN pip install --upgrade pip \
    && pip install -e .

# Expose port for HF Spaces
EXPOSE 7860

# 🔥 Use OpenEnv entrypoint (NOT uvicorn directly)
CMD ["server"]