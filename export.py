from ultralytics import YOLO

# Load a model
model = YOLO('/home/adminn/桌面/ultralytics-main/Hand-pose/PoseDetect_s/weights/best.pt')  # load a custom trained model

# Export the model
model.export(format='onnx')