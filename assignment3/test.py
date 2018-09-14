import torch


if __name__ == "__main__":
    a = {}
    a= torch.load('mytraining.pth')
    print(a.keys())
