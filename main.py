import asyncio
import gc
from fastapi import FastAPI, File, UploadFile,HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import gdown

# ========== Download Models ==========
def download_models():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    model_files = {
        "amharic_yolov8n.pt": "17CUOjC-s0iWKAZWIwRbOLTfFMe_nT687",
        "amharic_best.pth": "1yehU6v6_CnX05xfgtOwDl0Za1FtXuUoI",
        "vocab.txt": "1xh5fwLlE0oAhdP0HFGMfDB0rXR4Cdgz2"
    }

    for filename, file_id in model_files.items():
        output_path = os.path.join(model_dir, filename)
        if not os.path.exists(output_path):
            print(f"📥 Downloading {filename}...")
            gdown.download(id=file_id, output=output_path, quiet=False)
        else:
            print(f"✅ {filename} already exists.")

download_models()  # Only one call and one definition

# ========== Setup ==========
app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== CRNN Model ==========
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0),
        )
        self.lstm = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv(x).squeeze(2).permute(2, 0, 1)
        x, _ = self.lstm(x)
        return torch.nn.functional.log_softmax(self.fc(x), dim=2)

# ========== Load Models ==========
model_dir = "models"
yolo_model = YOLO(os.path.join(model_dir, "amharic_yolov8n.pt")).half()

vocab_path = os.path.join(model_dir, "vocab.txt")
vocab = [line.strip() for line in open(vocab_path, encoding="utf-8").readlines()]
if "<blank>" not in vocab:
    vocab.append("<blank>")
if " " not in vocab:
    vocab.append(" ")

char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for char, idx in char2idx.items()}

crnn = CRNN(num_classes=len(vocab)).half()
crnn.load_state_dict(torch.load(os.path.join(model_dir, "amharic_best.pth"), map_location=device))
crnn.to(device).eval()

# ========== Image Preprocessing ==========
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
])

# ========== Helper Functions ==========
def decode_prediction(prediction):
    pred_indices = prediction.argmax(2).permute(1, 0)
    decoded_texts = []
    for indices in pred_indices:
        decoded = []
        prev_idx = -1
        for idx in indices:
            idx = idx.item()
            if idx != prev_idx and idx != char2idx['<blank>']:
                decoded.append(idx2char[idx])
            prev_idx = idx
        decoded_texts.append(''.join(decoded))
    return decoded_texts

def sort_boxes(boxes):
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    lines, current_line = [], []
    line_thresh = 10
    for box in boxes:
        if not current_line or abs(box[1] - current_line[-1][1]) < line_thresh:
            current_line.append(box)
        else:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [box]
    if current_line:
        lines.append(sorted(current_line, key=lambda b: b[0]))
    return lines

def recognize_text(image_path):
    print(f"🟠 Processing {image_path}")  # Debug
    try:
         # Verify file exists
        if not os.path.exists(image_path):
            print("🔴 [A1] File not found")
            return None
            
        # Verify image can be read
        img = cv2.imread(image_path)
        if img is None:
            print("🔴 [A2] Failed to read image")
            return None
            
        print("🚀 Running inference on:", image_path)
        results = yolo_model(image_path, conf=0.25, max_det=1000 , imgsz=320)[0]
        print("✅ YOLO inference complete.")
        boxes = results.boxes.xyxy.cpu().numpy()
        print("🧱 Boxes detected:", len(boxes))
        img = cv2.imread(image_path)
        if img is None:
            return "❌ Failed to read image."
        word_boxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in boxes]

        sorted_lines = sort_boxes(word_boxes)
        final_text = []

        for line in sorted_lines:
            line_words, prev_x2 = "", None
            for x1, y1, x2, y2 in line:
                crop = img[y1:y2, x1:x2]
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                input_tensor = transform(pil_crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = crnn(input_tensor)
                text = decode_prediction(output)[0]
                if prev_x2 is not None:
                    gap = x1 - prev_x2
                    spaces = max(1, int(gap / 10))
                    line_words += " " * spaces
                line_words += text
                prev_x2 = x2
            final_text.append(line_words)
        return "\n".join(final_text)
        if not final_text:  # Add validation
                print("🟡 Warning: Empty recognition result")
                return "No text detected"
                
        print(f"🟢 Recognized text: {final_text}")
        return final_text
        
    except Exception as e:
        print(f"🔴 Recognition error: {str(e)}")
        return None

# ========== Endpoint ==========
from fastapi.responses import PlainTextResponse
@app.post("/ocr", response_class=PlainTextResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    print("🔵 [1/5] Endpoint called")  # Debug 1
    temp_path = None
    try:
        # Verify file received
        print(f"🔵 [2/5] Received file: {file.filename}, {file.size} bytes")  # Debug 2
        
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"🔵 [3/5] Saved to {temp_path}")  # Debug 3
        
        # Process with timeout
        try:
            print("🔵 [4/5] Starting recognition...")  # Debug 4
            text = await asyncio.wait_for(
                asyncio.to_thread(recognize_text, temp_path),
                timeout=110
            )
            print(f"🔵 [5/5] Recognition complete: {text}")  # Debug 5
            
            # Return as plain text (not dict!)
            return text if text else "No text detected"
            
        except asyncio.TimeoutError:
            return "Error: Processing timeout"
            
    except Exception as e:
        print(f"🔴 ERROR: {str(e)}")  # Debug error
        return f"Error: {str(e)}"
        
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        print("🟣 Cleanup complete")  # Debug cleanup
# Add this at the VERY BOTTOM of the file
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
