import torch  
import torchvision.transforms as transforms  
from PIL import Image  
from torchvision.models import resnet50  
import torch.nn.functional as F  
# 类别名称映射  
categories = ['airplane','airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud']  
# categories = ['aGrass','bField', 'cIndustry', 'dRiverLake', 'eForest', 'fResident', '00gParking6', 'gParking']  
# 加载预训练的ResNet50模型  
model_path = '/root/WYH/DA-Alone-Improves-AT-main/results/NWPU/resnet50_training/best_train_model.pth'  
resnet50_model = resnet50(weights=None)    # 不使用torchvision的预训练权重  
resnet50_model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))  # 加载你的模型权重  
resnet50_model.eval()  # 设置为评估模式  
  
# 确保模型在正确的设备上  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
resnet50_model = resnet50_model.to(device)  
  
# 定义一个图片预处理流程  
transform = transforms.Compose([  
    transforms.Resize(256),  
    transforms.CenterCrop(224),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])  
# 加载并处理图片  
# input_image_path = '/root/WYH-Project/Attack_competition/attack_dataset/clean_samples/006/00187997.jpg' 
input_image_path = '/root/WYH/data/NWPU/train/airport/airport_001.jpg' 
try:  
    input_image = Image.open(input_image_path)  
    input_tensor = transform(input_image).unsqueeze(0).to(device)  # 增加一个批次维度并移动到设备 
    # 前向传播  
    with torch.no_grad():  
        outputs = resnet50_model(input_tensor)  
    # 获取预测结果  
    _, predicted = torch.topk(outputs, 2)  # 获取置信度最高的两个标签  
    # print(predicted.shape)  
    predicted_index = predicted[0, 0].item()  # 获取第一个样本的最大概率索引的标量值
    # print(predicted_index)
    # 获取置信度  
    softmax_outputs = F.softmax(outputs, dim=1)  
    top1_index = predicted[0, 0]  
    top2_index = predicted[0, 1]  
# 使用索引从 softmax 输出中获取相应的置信度  
    top1_confidence = softmax_outputs[0, top1_index].item()  
    top2_confidence = softmax_outputs[0, top2_index].item()  
# 输出预测结果  
    predicted_classes = [categories[idx.item()] for idx in predicted[0]]  # 注意这里使用 predicted[0]  
    print(f'Predicted Top 1 class: {predicted_classes[0]}, Confidence: {top1_confidence:.4f}')  
    print(f'Predicted Top 2 class: {predicted_classes[1]}, Confidence: {top2_confidence:.4f}')

except Exception as e:  
    print(f"An error occurred: {predicted_index}")