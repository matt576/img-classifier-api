import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load model
# model = models.resnet18(pretrained=True)
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load class labels
with open("imagenet_classes.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def predict_image(image: Image.Image):
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
    _, index = torch.max(outputs, 1)
    return labels[index.item()]