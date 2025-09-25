# MNIST Digit Classifier (FastAPI + Streamlit + PyTorch)

This project is a full-stack example of serving a deep learning model for digit recognition on the MNIST dataset.  
It combines:

- **PyTorch**: CNN model for digit classification.  
- **FastAPI**: Backend API for inference.  
- **Streamlit**: Simple frontend to upload and test images.  
- **Docker**: Containerized deployment for portability.  

---

## Project Structure

```
MNIST_Classifier/
│
├── CNN_class.py              # PyTorch CNN model definition
├── MNIST_app.py              # FastAPI backend (serving model predictions)
├── MNIST_streamlit_app.py    # Streamlit frontend (upload + predict UI)
├── mnist_cnn.pth             # Trained model weights
├── requirements.txt          # Dependencies
├── Dockerfile                # Docker setup for containerized deployment
├── data/                     # (excluded from container) raw training data
└── MNIST.ipynb               # (excluded from container) training notebook
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/MNIST_Classifier.git
cd MNIST_Classifier
```

### 2. Install dependencies (local run)
```bash
pip install --no-cache-dir -r requirements.txt
```

### 3. Run FastAPI backend
```bash
uvicorn MNIST_app:app --host 0.0.0.0 --port 8000
```
- API docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Run Streamlit frontend
```bash
streamlit run MNIST_streamlit_app.py
```
- UI available at: [http://localhost:8501](http://localhost:8501)

---

## Running with Docker

### 1. Build image
```bash
docker build -t mnist-app .
```

### 2. Run container
```bash
docker run -it -p 8000:8000 -p 8501:8501 mnist-app
```

### 3. Access services
- FastAPI → [http://localhost:8000/docs](http://localhost:8000/docs)  
- Streamlit → [http://localhost:8501](http://localhost:8501)  

---

## Pre-Built Container

A pre-built Docker image is available on **GitHub Container Registry (GHCR)**:  

```bash
docker pull ghcr.io/anakin-sudo/mnist_application:latest
docker run -it -p 8000:8000 -p 8501:8501 ghcr.io/<your-username>/mnist-app:latest
```
---

## API Usage

Example request (with `curl`):
```bash
curl -X POST "http://localhost:8000/predict/"   -H "accept: application/json"   -H "Content-Type: multipart/form-data"   -F "file=@digit.png"
```

Response:
```json
{
  "predicted_class": 7,
  "probabilities": [0.01, 0.00, 0.02, 0.00, 0.03, 0.00, 0.90, 0.02, 0.01, 0.01]
}
```
## Frontend Usage (Streamlit)
Open [http://localhost:8501](http://localhost:8501) in your browser, and you can:

- Upload an image of a handwritten digit.  
- View the predicted class.  
- See the probability distribution across all 10 digits.  

This provides a graphical interface to explore the model, without needing to send API requests manually.


---

## Notes
- The training notebook (`MNIST.ipynb`) and raw data are excluded from the container.  
- The container only includes **inference-related files** (`CNN_class.py`, `MNIST_app.py`, `MNIST_streamlit_app.py`, `mnist_cnn.pth`).  
- For lighter Docker builds, we use CPU-only PyTorch wheels. GPU support can be enabled if needed.  

---
