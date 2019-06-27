import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# DATADIR = os.path.dirname(__file__) + '/../data'
# DATADIR = os.path.dirname(__file__) + '/data'
DATADIR = './data'
print(DATADIR)
BATCH_SIZE = 128


class LabelCorrupter(torch.utils.data.Dataset):
    def __init__(self, dataset, corruption_chance: float = 1.0, seed: int = 42):
        self.dataset = dataset
        self.random_labels = np.random.choice(np.unique(self.dataset.targets).tolist(),
                                              size=len(self.dataset))
        self.corrupt_labels = torch.from_numpy(self.create_corrupt_labels(corruption_chance, seed))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item][0], self.corrupt_labels[item]

    def create_corrupt_labels(self, corruption_chance: float = 1.0, seed: int = 42):
        np.random.seed(seed)
        random_labels = np.random.choice(np.unique(self.dataset.targets).tolist(), size=len(self.dataset))
        corruption_mask = np.random.rand(len(self.dataset)) <= corruption_chance
        not_corrupt_mask = np.logical_not(corruption_mask)

        assert np.all(corruption_mask + not_corrupt_mask)
        assert len(corruption_mask) == len(self.dataset.targets)
        assert len(not_corrupt_mask) == len(self.dataset.targets)

        return not_corrupt_mask * self.dataset.targets.numpy() + corruption_mask * random_labels


def fashion_mnist(corruption_chance=0.0, batch_size=4):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize([0.5], [0.5])])

    train_set = torchvision.datasets.FashionMNIST(root=DATADIR, train=True, download=True, transform=transform)
    train_set_corrupt = LabelCorrupter(train_set, corruption_chance)
    print(f"Percentage corrupt: "
          f"{torch.tensor(1.0) - torch.mean(torch.eq(train_set_corrupt.corrupt_labels, train_set.targets).double())}")
    train_loader = torch.utils.data.DataLoader(train_set_corrupt, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.FashionMNIST(root=DATADIR, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    return train_loader, test_loader, classes


def mnist(corruption_chance=0.0, batch_size=4):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize([0.5], [0.5])])

    train_set = torchvision.datasets.MNIST(root=DATADIR, train=True, download=True, transform=transform)
    train_set_corrupt = LabelCorrupter(train_set, corruption_chance)
    print(f"Percentage corrupt: "
          f"{torch.tensor(1.0) - torch.mean(torch.eq(train_set_corrupt.corrupt_labels, train_set.targets).double())}")
    train_loader = torch.utils.data.DataLoader(train_set_corrupt, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.MNIST(root=DATADIR, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

    return train_loader, test_loader, classes


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(28 * 28, 512)
        self.output = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x


if __name__ == '__main__':
    corruption = np.linspace(0.0, 1.0, num=5)

    models = []
    mlp = MLP()

    # for corrupt in corruption:
    for corrupt in [corruption[-1]]:
        # trainloader, testloader, _ = mnist(corrupt, batch_size=BATCH_SIZE)
        trainloader, testloader, _ = fashion_mnist(corrupt, batch_size=BATCH_SIZE)

        # mlp.apply(weights_init)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        counter = 0
        losses = []

        disp_batch = len(trainloader) // 5
        # n_epochs = int(np.ceil(20000/len(trainloader)))
        n_epochs = 600

        for epoch in range(n_epochs):
            print(f"LR = {optimizer.param_groups[0]['lr']}")
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = mlp(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % disp_batch == disp_batch - 1:  # print 4 times
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / disp_batch))
                    running_loss = 0.0

                losses.append(loss.item())
                counter += 1

            scheduler.step()

        models.append(losses)
        print('Finished Training')

    models = np.asarray(models)
    np.save('models.npy', models)
    # plt.plot([a.mean() for a in np.split(np.asarray(losses), len(losses) / 1000)])
    # plt.show()
