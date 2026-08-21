# Base image
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app, model, pipeline and artifacts code used
COPY app/ ./app/
COPY model/ ./model/
COPY pipeline/ ./pipeline/
COPY artifacts/ ./artifacts/

# Expose port Streamlit will run on
EXPOSE 8501

# Runtime command
CMD streamlit run app/homepage.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true