from PIL import Image
import torch.nn as nn
import torch
from visiontransformer import VisionTransformer
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def predict(model, img_tensor):
    model.eval()
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

    return predicted

model = VisionTransformer(
    d_model = 64,
    n_classes = 10,
    image_size = (32,32),
    patch_size = (8,8),
    n_channels = 1,
    n_heads = 4,
    n_layers = 3
)

model.load_state_dict(torch.load("ViTr_MNIST.pth", map_location=device))

transform = transforms.Compose([
    transforms.Grayscale(),              
    transforms.Resize((32, 32)),         
    transforms.ToTensor()
])

img = Image.open('image3.png')
img_tensor = transform(img).unsqueeze(0)

predicted_number = predict(model, img_tensor)
print(predicted_number)