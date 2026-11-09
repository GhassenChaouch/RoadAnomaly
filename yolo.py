import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# ==========================================
# 1. LOAD MODELS
# ==========================================
# Load YOLO for common objects (Pedestrians, Cars)
yolo_model = YOLO('yolov8s.pt') 

# Load custom model from the deeplearning Anomaly project
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = YourModelClass() 
# model.load_state_dict(torch.load("model_weights.pth", map_location=device))
# model.to(device)
# model.eval()

image_size = (256, 256)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# ==========================================
# 2. DEFINE DRIVING CORRIDOR (ROI)
# ==========================================
def get_roi_polygon(width, height):
    # Creates a trapezoid in the lower half of the screen
    top_left = (int(width * 0.3), int(height * 0.6))
    top_right = (int(width * 0.7), int(height * 0.6))
    bottom_right = (int(width * 0.9), height)
    bottom_left = (int(width * 0.1), height)
    return np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)

cap = cv2.VideoCapture(0)
print("Starting Braking System. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    height, width = frame.shape[:2]
    original_frame = frame.copy()
    trigger_brake = False
    brake_reason = ""
    
    # Draw the ROI (Driving Corridor) in faint blue
    roi_poly = get_roi_polygon(width, height)
    cv2.polylines(frame, [roi_poly], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # ------------------------------------------
    # A. YOLO PASS (Standard Objects)
    # ------------------------------------------
    results = yolo_model.predict(frame, verbose=False)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        
        # Calculate the bottom-center point of the object (where it touches the ground)
        bottom_center = (int((x1 + x2) / 2), y2)
        
        # Check if this point is inside the driving corridor
        if cv2.pointPolygonTest(roi_poly, bottom_center, False) >= 0:
            trigger_brake = True
            brake_reason = f"{class_name} in path!"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # Red box
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box

    # ------------------------------------------
    # B. PYTORCH ANOMALY PASS (Rare Hazards)
    # ------------------------------------------
    rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # NOTE: Uncomment when your model weights are ready
    """
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        
    mask_resized = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Check if anomaly mask overlaps with the ROI polygon
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_poly], 1)
    
    # Bitwise AND to find anomaly pixels inside the driving corridor
    overlap = cv2.bitwise_and(mask_resized, roi_mask)
    
    # If more than 500 anomaly pixels are in the path, trigger brake
    if np.sum(overlap) > 500:
        trigger_brake = True
        brake_reason = "ANOMALY: Hazard in path!"
        
    # Overlay the red anomaly mask on the screen
    overlay = frame.copy()
    overlay[mask_resized == 1] = [0, 0, 255]
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    """

    # ------------------------------------------
    # C. BRAKING ALERT SYSTEM
    # ------------------------------------------
    if trigger_brake:
        # Flash the screen red and display large text
        alert_overlay = frame.copy()
        alert_overlay[:] = (0, 0, 255)
        frame = cv2.addWeighted(alert_overlay, 0.3, frame, 0.7, 0)
        
        cv2.putText(frame, "EMERGENCY BRAKE!", (50, 100), 
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)
        cv2.putText(frame, brake_reason, (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "PATH CLEAR", (50, 100), 
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 5)

    cv2.imshow("Active Braking System", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()