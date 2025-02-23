import torchvision
from torch.utils import data
from torchvision import transforms

def load_data(batch_size):
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))