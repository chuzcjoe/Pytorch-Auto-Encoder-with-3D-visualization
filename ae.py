import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from utils import *

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1_layer_1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=(1, 1)
        )
        self.conv1_layer_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=(1, 1)
        )
        self.conv2_layer_1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=(1, 1)
        )
        self.conv2_layer_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=(1, 1)
        )
        self.linear1 = nn.Linear(64*492*492, 3)

    def forward(self, features):
        activation = self.conv1_layer_1(features)
        activation = F.relu(activation)
        activation = self.conv1_layer_2(activation)
        activation = F.relu(activation)
        activation = self.conv2_layer_1(activation)
        activation = F.relu(activation)
        activation = self.conv2_layer_2(activation)
        code = F.relu(activation)
        #add
        code = code.view(code.size()[0], -1)
        code = self.linear1(code)
        return code


class Decoder(nn.Module):
    def __init__(self, ob):
        super(Decoder, self).__init__()
        self.convt1_layer_1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=(1, 1)
        )
        self.convt1_layer_2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=(1, 1)
        )
        self.convt2_layer_1 = nn.ConvTranspose2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=(1, 1)
        )
        self.convt2_layer_2 = nn.ConvTranspose2d(
            in_channels=32, out_channels=3, kernel_size=3, stride=(1, 1)
        )

        self.linear2 = nn.Linear(3, 64*492*492)

    def forward(self, features):
        #add
        activation = self.linear2(features)
        activation = activation.view(-1, 64, 492, 492)
        activation = self.convt1_layer_1(activation)
        activation = F.relu(activation)
        activation = self.convt1_layer_2(activation)
        activation = F.relu(activation)
        activation = self.convt2_layer_1(activation)
        activation = F.relu(activation)
        activation = self.convt2_layer_2(activation)
        reconstructed = F.relu(activation)
        return reconstructed


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(self.encoder)

    def forward(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed


class TrainDataSet(Dataset):
    def __init__(self, data_dir, transform, image_mode="RGB"):
        self.data_dir = data_dir
        self.transform = transform
        self.image_mode = image_mode
        self.data_list = os.listdir(os.path.join(self.data_dir, 'imgs'))
        self.length = len(self.data_list)

    def __getitem__(self, index, crop=True):
        base_name, _ = self.data_list[index].split('.')

        #read img
        img = Image.open(os.path.join(self.data_dir, 'imgs/'+base_name+'.jpg'))
        img = img.convert(self.image_mode)

        if crop:
            #get bbox
            bbox = Bbox(os.path.join(self.data_dir, 'labels/'+base_name+'.txt'))
            x_min, x_max, y_min, y_max = bbox

            img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        if self.transform:
            img = self.transform(img)

        return img
    
    def __len__(self):
        return self.length


transform = torchvision.transforms.Compose([
    transforms.Resize((500,500)),
    transforms.ToTensor()
    ])


train_dataset = TrainDataSet('./', transform)
print("Traning samples:", train_dataset.length)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)
model = AE().cuda(0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss().cuda(0)

print(model)
print([parameter.size() for parameter in model.parameters()])

epochs = 20
print("start training...")
cost = float('inf')
for epoch in range(epochs):
    print('epoch:', epoch)
    loss = 0
    for batch_features in train_loader:
        batch_features = batch_features.cuda(0)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs, batch_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
    loss = loss / len(train_loader)
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
    if loss < cost:
        cost = loss
        print('Better model')
        print('Saving...')
        torch.save(model.state_dict(), 'best.pkl')
        print('Saved!')

#with torch.no_grad():
#    number = 10
#    plt.figure(figsize=(20, 4))
#    for index in range(number):
#        # display original
#        ax = plt.subplot(2, number, index + 1)
#        plt.imshow(test_dataset.data[index].reshape(28, 28))
#        plt.gray()
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)

        # display reconstruction
#        ax = plt.subplot(2, number, index + 1 + number)
#        test_data = test_dataset.data[index]
#        test_data = test_data.to(device)
#        test_data = test_data.float()
#        test_data = test_data.view(-1, 1, 28, 28)
#        output = model(test_data)
#        plt.imshow(output.cpu().reshape(28, 28))
#        plt.gray()
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
#    plt.show()
