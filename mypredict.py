from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(r'/home/adminn/Desktop/ultralytics-main/Hand-pose-pck/rle4/weights/best.pt')

# Run inference on 'bus.jpg' with arguments
model.predict(r'/home/adminn/Desktop/ultralytics-main/ultralytics/datasets3/images/val', save=True, imgsz=640, conf=0.5, visualize=False)