# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only inference-related files
COPY CNN_class.py .
COPY MNIST_app.py .
COPY MNIST_streamlit_app.py .
COPY mnist_cnn.pth .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default command: start FastAPI + Streamlit
CMD uvicorn MNIST_app:app --host 0.0.0.0 --port 8000 & \
    streamlit run MNIST_streamlit_app.py --server.address 0.0.0.0 --server.port 8501
