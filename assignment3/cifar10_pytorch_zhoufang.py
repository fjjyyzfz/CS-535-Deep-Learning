from __future__ import print_function
from __future__ import division
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import torch
from collections import defaultdict
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        #self.fc1 = nn.Linear(64 * 8 * 8, 10)
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.batch = nn.BatchNorm1d(512)
        #self.fcadd = nn.Linear(512, 512)
        #self.fc2 = nn.Linear(512, 10)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        #x = F.relu(self.fc1(x))
        x = self.batch(self.fc1(x))
        #x = F.dropout(x, 0.5)
        x = F.relu(x)
        #x = F.relu(self.fcadd(x))
        x = self.fc2(x)
        #x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() # Why would I do this?
    criterion = nn.CrossEntropyLoss(size_average=False)
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.data[0]
    net.train() # Why would I do this?
    return total_loss / total, correct / total

def drawPlotLoss(num_epochs, test_acc_out, traintest, parameter, number):
    """
    in drawPlot function, we plot the test accuracy with different parameters and save it.
    """
    plt.figure()
    plt.plot(range(1,num_epochs + 1), test_acc_out[0], '-bo', label = "%s %s" %(parameter[0], traintest))
    plt.plot(range(1,num_epochs + 1), test_acc_out[1], '-rs', label = "%s %s" %(parameter[1], traintest))
    plt.xlim(1, num_epochs)
    plt.ylim(0, 1.3)
    
    plt.xlabel("Epoch")
    plt.ylabel("%s" %(traintest))
    plt.grid(True)
    plt.legend(loc = 'lower left')
    plt.title("%s of %s and %s" %(traintest, parameter[0], parameter[1]))
    plt.savefig("%s_%d.png" %(traintest, number))

def drawPlotAcc(num_epochs, test_acc_out, traintest, parameter, number):
    """
    in drawPlot function, we plot the test accuracy with different parameters and save it.
    """
    plt.figure()
    plt.plot(range(1,num_epochs + 1), test_acc_out[0], '-bo', label = "%s %s" %(parameter[0], traintest))
    plt.plot(range(1,num_epochs + 1), test_acc_out[1], '-rs', label = "%s %s" %(parameter[1], traintest))
    plt.xlim(1, num_epochs)
    plt.ylim(0.65, 1.1)
    
    plt.xlabel("Epoch")
    plt.ylabel("%s" %(traintest))
    plt.grid(True)
    plt.legend(loc = 'lower right')
    plt.title("%s of %s and %s" %(traintest, parameter[0], parameter[1]))
    plt.savefig("%s_%d.png" %(traintest, number))

if __name__ == "__main__":
    BATCH_SIZE = 32 #mini_batch size
    MAX_EPOCH = 30 #maximum epoch to train

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    #print(list(trainset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model...')


    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    net = Net().cuda()
    net.train() # Why would I do this?
    # pretrain_dict = {}
    # model_dict = {}
    # new_model = {}
    # pretrain_dict = torch.load('mytraining.pth')
    # model_dict = net.state_dict()
    # #print(pretrain_dict)
    # for key in model_dict.keys():
    #     if 'fc2' in key or 'fcadd' in key:
    #         new_model[key] = model_dict[key]
    #     else:
    #         new_model[key] = pretrain_dict[key]

    # net.load_state_dict(new_model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)
    #optimizer = optim.Adagrad(net.parameters(), lr=0.01)
    #optimizer = optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.999))
    #optimizer = optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99)
    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            #print(list(inputs))
            print(labels)
            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        test_loss, test_acc = eval_net(testloader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))
    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), 'mytraining4.pth')
    y1 = []
    y1.append(train_loss_list)
    y1.append(test_loss_list)
    drawPlotLoss(MAX_EPOCH, y1, 'Loss', ['Train', 'Validation'], 43)
    y2 = []
    y2.append(train_acc_list)
    y2.append(test_acc_list)
    drawPlotAcc(MAX_EPOCH, y2, 'Accuracy', ['Train', 'Validation'], 43)
    