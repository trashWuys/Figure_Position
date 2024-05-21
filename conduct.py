import torch
from torchvision import transforms
from PIL import Image
import Model


weight = 'weight_file'
image_path = 'image_file'


# 重新定义图像
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载并处理图像
image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # Add batch dimension

# 判断使用GPU还是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = Model.MyModel().to(device)

# 加载权重文件
model.load_state_dict(torch.load(weight))
model.eval()
print("Model weights loaded from model_weights.pth")

# 预测
with torch.no_grad():
    image = image.to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)  # Assuming classification task
    predicted_class = predicted.item()

# 打印结果
print(f"Predicted class: {predicted_class}")
