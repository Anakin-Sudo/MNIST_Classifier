from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
import io
import torch
import torchvision.transforms as transforms
from CNN_class import CNN

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model once at startup
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu", weights_only=True))
model.to(device)
model.eval()

# Same transform as training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # force grayscale
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    # Preprocess
    img_tensor = transform(img).unsqueeze(0)  # shape [1, 1, 28, 28]
    img_tensor = img_tensor.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()

    return {
        "predicted_class": int(pred_class),
        "probabilities": [round(p, 4) for p in probs.squeeze().tolist()]
    }
