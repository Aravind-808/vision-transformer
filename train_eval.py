import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from visiontransformer import VisionTransformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#params
d_model = 64
n_classes = 10
img_size = (32,32)
patch_size = (8,8)
n_channels = 1
n_heads = 4
n_layers = 3
batch_size = 128
epochs = 10
alpha = 0.001

model = VisionTransformer(
    d_model,
    n_classes,
    img_size,
    patch_size,
    n_channels,
    n_heads,
    n_layers
).to(device)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=128)

optimizer = Adam(model.parameters(), lr = alpha)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for img, label in train_loader:
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
    
    print(f"Epoch: {epoch+1}, Loss: {train_loss:.3f}")

# model.eval()
# correct = 0
# with torch.no_grad():
#     for img, label in test_loader:
#         img, label = img.to(device), label.to(device)
#         output = model(img)
#         _, predicted = torch.max(output, 1)
#         correct += (predicted == label).sum().item()

# print(f"Test Accuracy: {100 * correct / len(test_data):.2f}%")

torch.save(model.state_dict(), "ViTr_MNIST.pth")