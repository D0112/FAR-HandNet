from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('/home/adminn/Desktop/ultralytics-main/ultralytics/cfg/models/v8/Hand-pose-Fla-RLE.yaml')

# Load a pretrained YOLO model (recommended for training)


# model = YOLO('D:/pythonProject/ultralytics-main/yolov8n-pose.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='/home/adminn/桌面/ultralytics-main/ultralytics/cfg/datasets/L-Hand-pose.yaml',
#                       epochs=300,batch=128,pose=12,project='L-Hand-pose',name='128-Baseline',pretrained=False,device=[0,1],resume=True)
#
results = model.train(data='/home/adminn/Desktop/ultralytics-main/ultralytics/cfg/datasets/Hand-pose-CMU.yaml',
                          epochs=30,batch=32,pose=12,project='Hand-pose',name='resnet34',pretrained=False,device=[0], single_cls=True, mosaic=1.0, split='test')

# results = model.train(data='/home/adminn/桌面/ultralytics-main/ultralytics/cfg/datasets/test.yaml',
#                       epochs=20, batch=64, pose=12, project='Hand-test', name='Fla_dsc', pretrained=False, device=[0,1], workers=0, mosaic=0.0)

# resume
# model = YOLO('/home/adminn/桌面/ultralytics-main/Hand-pose/rle_ex_s3/weights/last.pt')
# results = model.train(resume=True)
# Evaluate the model's performance on the validation set
results = model.val()

# # Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')
#
# # Export the model to ONNX format
# success = model.export(format='onnx')
