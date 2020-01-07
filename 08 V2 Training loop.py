import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as op

import torchvision
import torchvision.transforms as tfs

print(torch.__version__)
print(torchvision.__version__)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=192, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 192)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
        return t


train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    , download=True
    , transform=tfs.Compose([tfs.ToTensor()])
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)

nw1 = Network()
optimizer = op.Adam(nw1.parameters(), lr=0.01)

for epoch in range(0, 5):
    total_loss = 0
    total_correct = 0

    for batch in train_loader:
        images, labels = batch
        preds = nw1(images)

        loss = F.cross_entropy(preds, labels)
        optimizer.zero_grad()

        loss.backward()  # Calculate the gradients
        optimizer.step()  # Update the weights
        # print(loss.item(), get_num_correct(preds, labels))
        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print('Epoch', epoch,
          'Successful predication rate: {:2.2%}'.format(total_correct / len(train_set)),
          'Total loss: ', total_loss)
