"""
CS 535 Deep Learning Final Project
Author: Zhou Fang,Jiale Liu 
Final Update: 03/20/2018
Discription: This project is based on UCI HAR Dataset "Human Activity recognition 
with smartphone" and run code in GPU by pytorch. MLP and LTSM are performed 
and performance are compared between these two algorithms. Improvement are done in feature selection:
Importance in randomforestclassifier, Mutual imformation feature selection, F-score.
"""
from __future__ import print_function
from __future__ import division
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier

class MLPNet(nn.Module):
    """
    in MLPNet class, MLP model is defined
    """
    def __init__(self, num_feature, hidden_dim, batch_size):
        super(MLPNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.fc1 = nn.Linear(num_feature, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 6)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        return 512
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LSTMNet(nn.Module):
    """
    in LSTMNet class, LSTM model is defined
    """
    def __init__(self, num_feature, hidden_dim, batch_size):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_feature = num_feature
        self.lstm = nn.LSTM(num_feature, hidden_dim, dropout = 0.5)
        self.fc = nn.Linear(hidden_dim, 6)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        return (h0, c0)
    def forward(self, x):
        x = x.view(-1, self.batch_size, self.num_feature)
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.fc(x[-1])
        return x

def eval_net(data_x_loader, data_y_loader, total_dict, net):
    """
    in eval_net function, evaluate model performance, return loss, accuracy,
    6 activities accuracy and model
    """
    correct = 0
    correct_dict = defaultdict(int)
    acc_dict = defaultdict(int)
    prediction = []
    total = 0
    total_loss = 0
    net.eval() 
    criterion = nn.CrossEntropyLoss(size_average=False)
    for i, (params, labels) in enumerate(zip(data_x_loader, data_y_loader)):
        params, labels = (Variable(params).float()).cuda(), Variable(labels).cuda()
        net.batch_size = len(labels)
        net.hidden = net.init_hidden()
        outputs = net(params)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        for i, predict in enumerate(predicted):
            prediction.append(predict)
            if predict == labels.data[i]:
                correct_dict[predict] += 1
                correct += 1
        loss = criterion(outputs, labels)
        total_loss += loss.data[0]
    for key, item in correct_dict.items():
            acc_dict[key] = item/total_dict[key]
    net.train()
    return total_loss / total, correct / total, acc_dict, prediction, net

def WriteResult(file, epoch, train_loss, train_acc, test_loss, test_acc):
    """
    in WriteResult function, write train loss, test loss, train acc, test acc in each epoch
    """
    file.write('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f\n' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))

def WriteDictResult(file, epoch, classes, acc_dict):
    """
    in WriteDictResult function, write test loss in 6 activities in each epoch
    """
    file.write('EPOCH: %d' %(epoch+1))
    for i, item in enumerate(classes):
        file.write(' %s: %.5f' %(item, acc_dict[i]))
    file.write('\n')

def drawPlotLoss(num_epochs, loss_out, parameter, resultlist, parameter2):
    """
    in drawPlotLoss function, we plot the train and test loss with model and save it.
    """
    plt.figure()
    plt.plot(range(1,num_epochs + 1), loss_out[0], '-bo', label = "%s %s" %(resultlist[0], parameter))
    plt.plot(range(1,num_epochs + 1), loss_out[1], '-yv', label = "%s %s" %(resultlist[1], parameter))
    plt.plot(range(1,num_epochs + 1), loss_out[2], '-cd', label = "%s %s" %(resultlist[2], parameter))
    plt.plot(range(1,num_epochs + 1), loss_out[3], '-rs', label = "%s %s" %(resultlist[3], parameter))
    plt.xlim(1, num_epochs)
    plt.ylim(0, 1.5)
    plt.xlabel("Epoch")
    plt.ylabel("%s" %(parameter))
    plt.grid(True)
    plt.legend(loc = 'upper right')
    plt.title("%s of %s and %s" %(parameter, parameter2[0], parameter2[1]))
    plt.savefig(("%s of %s and %s.png" %(parameter, parameter2[0], parameter2[1])))

def drawPlotAcc(num_epochs, acc_out, parameter, resultlist, parameter2):
    """
    in drawPlotAcc function, we plot the train and test accuracy with different model and save it.
    """
    plt.figure()
    plt.plot(range(1,num_epochs + 1), acc_out[0], '-bo', label = "%s %s" %(resultlist[0], parameter))
    plt.plot(range(1,num_epochs + 1), acc_out[1], '-yv', label = "%s %s" %(resultlist[1], parameter))
    plt.plot(range(1,num_epochs + 1), acc_out[2], '-cd', label = "%s %s" %(resultlist[2], parameter))
    plt.plot(range(1,num_epochs + 1), acc_out[3], '-rs', label = "%s %s" %(resultlist[3], parameter))
    plt.xlim(1, num_epochs)
    plt.ylim(0.5, 1.0)
    plt.xlabel("Epoch")
    plt.ylabel("%s" %(parameter))
    plt.grid(True)
    plt.legend(loc = 'lower right')
    plt.title("%s of %s and %s" %(parameter, parameter2[0], parameter2[1]))
    plt.savefig(("%s of %s and %s.png" %(parameter, parameter2[0], parameter2[1])))

def drawPlotDict(num_epochs, acc_out, classes, model):
    """
    in drawPlotDict function, we plot the test accuracy with 6 activities and save it.
    """
    color_list = ['-go', '-rs', '-bp', '-cd', '-k*', '-yv']
    plt.figure()
    for i, parameter in enumerate(classes):
        plt.plot(range(1,num_epochs + 1), acc_out[i], color_list[i], label = "%s Accuracy" %(str(parameter)))
        plt.xlim(0, num_epochs)
        plt.ylim(0.5, 1.01)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.grid(True)
    plt.legend(loc = 'lower right')
    plt.title("%s Test Accuracy With Different Activity" %(str(model)))
    plt.savefig("%s_test_accuracy_with_different_dctivity.png" %(str(model)))

def drawPlotBar(classes, list1, list2, parameter):
    """
    in drawPlotBar function, we plot 6 activities acc at best accuracy achieved epoch with different models.
    """
    plt.figure(figsize=(9, 7))
    x = list(range(6))
    width = 0.35
    plot1 = plt.bar(x, list1, width = width, color = '#0072BC', label = parameter[0])
    for i in range(6):
        x[i] = x[i] + width
    plot2 = plt.bar(x, list2, width = width, color = '#ED1C24', label = parameter[1])
    plt.xticks(x, classes, rotation = 10)
    plt.ylim(0,105)
    plt.ylabel("Accuracy(percentage)")
    plt.legend(loc = 'lower right')
    def add_labels(valuelist):
        for value in valuelist:
            height = value.get_height()
            plt.text(value.get_x() + value.get_width() / 2, height, height, ha='center', va='bottom')
            value.set_edgecolor('white')
    add_labels(plot1)
    add_labels(plot2)
    plt.title("Comparison Between %s And %s With Different Activity" %(parameter[0], parameter[1]))
    plt.savefig("comparison_between_%s_and_%s_with_different_activity.png" %(parameter[0], parameter[1]))

def figurePlot(MAX_EPOCH, classes, train_loss_MLP, test_loss_MLP, train_loss_LSTM,
    test_loss_LSTM, train_acc_MLP, test_acc_MLP,train_acc_LSTM,test_acc_LSTM,
    laying_acc_MLP, sitting_acc_MLP, standing_acc_MLP, walking_acc_MLP,
    walkingdown_acc_MLP, walkingup_acc_MLP, laying_acc_LSTM, sitting_acc_LSTM,
    standing_acc_LSTM, walking_acc_LSTM, walkingdown_acc_LSTM, walkingup_acc_LSTM, file,
    parameter1, parameter2, parameter3):
    """
    in figurePlot function, we plot the train and test accuracy with different model, 
    the train and test loss with different model, the test accuracy in 6 activities with different model
    """
    y1 = []
    y1.append(train_loss_MLP)
    y1.append(test_loss_MLP)
    y1.append(train_loss_LSTM)
    y1.append(test_loss_LSTM)
    drawPlotLoss(MAX_EPOCH, y1, parameter1[0], parameter2, parameter3)
    y2 = []
    y2.append(train_acc_MLP)
    y2.append(test_acc_MLP)
    y2.append(train_acc_LSTM)
    y2.append(test_acc_LSTM)
    drawPlotAcc(MAX_EPOCH, y2, parameter1[1], parameter2, parameter3)
    y3 = []
    y3.append(laying_acc_MLP)
    y3.append(sitting_acc_MLP)
    y3.append(standing_acc_MLP)
    y3.append(walking_acc_MLP)
    y3.append(walkingdown_acc_MLP)
    y3.append(walkingup_acc_MLP)
    drawPlotDict(MAX_EPOCH, y3, classes, parameter3[0])
    y4 = []
    y4.append(laying_acc_LSTM)
    y4.append(sitting_acc_LSTM)
    y4.append(standing_acc_LSTM)
    y4.append(walking_acc_LSTM)
    y4.append(walkingdown_acc_LSTM)
    y4.append(walkingup_acc_LSTM)
    drawPlotDict(MAX_EPOCH, y4, classes, parameter3[1])
    list1 = []
    max_index1 = test_acc_MLP.index(max(test_acc_MLP))
    file.write("%d\n" %(max_index1 + 1))
    list1.append(float('%.2f'%(laying_acc_MLP[max_index1]*100)))
    list1.append(float('%.2f'%(sitting_acc_MLP[max_index1]*100)))
    list1.append(float('%.2f'%(standing_acc_MLP[max_index1]*100)))
    list1.append(float('%.2f'%(walking_acc_MLP[max_index1]*100)))
    list1.append(float('%.2f'%(walkingdown_acc_MLP[max_index1]*100)))
    list1.append(float('%.2f'%(walkingup_acc_MLP[max_index1]*100)))
    list2 = []
    max_index2 = test_acc_LSTM.index(max(test_acc_LSTM))
    file.write("%d\n" %(max_index2 + 1))
    list2.append(float('%.2f'%(laying_acc_LSTM[max_index2]*100)))
    list2.append(float('%.2f'%(sitting_acc_LSTM[max_index2]*100)))
    list2.append(float('%.2f'%(standing_acc_LSTM[max_index2]*100)))
    list2.append(float('%.2f'%(walking_acc_LSTM[max_index2]*100)))
    list2.append(float('%.2f'%(walkingdown_acc_LSTM[max_index2]*100)))
    list2.append(float('%.2f'%(walkingup_acc_LSTM[max_index2]*100)))
    drawPlotBar(classes, list1, list2, parameter3)

def LoadData(train, test, FC_fscore, FC_mifs):
    """
    in LoadData function, we load train and test data and put them into dataloader.
    return 6 activities, count for 6 activities, train_x, test_x, train_y and test_y dataloader, also
     get dataloader for feature selection.
    """
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)
    train_data = shuffle(train_data)
    test_data  = shuffle(test_data)
    train_label = train_data['Activity']
    test_label = test_data['Activity']
    train_x = np.asarray(train_data.drop(['subject', 'Activity'], axis=1))
    test_x = np.asarray(test_data.drop(['subject', 'Activity'], axis=1))

    # 561 features dataloader
    train_x_loader = torch.utils.data.DataLoader(train_x, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=4)
    test_x_loader = torch.utils.data.DataLoader(test_x, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=4)

    # transfer classes into integer
    encoder = preprocessing.LabelEncoder()
    encoder.fit(train_label)
    classes = list(encoder.classes_)
    train_y = np.asarray(encoder.transform(train_label))
    test_y = np.asarray(encoder.transform(test_label))

    # get 561 features name
    x_columns = [x for x in train_data.columns if x not in ['subject', 'Activity']]

    # chosse feature by F-score
    file = open(FC_fscore, 'r')
    selected_index = file.read().split(' ')
    selected_index = map(int, selected_index[: -1])
    print("numbers of selected feature for F-score: %d" %(len(selected_index)))
    file.close()
    selected_feature = []
    for i in selected_index:
        selected_feature.append(x_columns[i])
    train_x = np.asarray(pd.DataFrame(train_data, columns=selected_feature))
    test_x = np.asarray(pd.DataFrame(test_data, columns=selected_feature))
    train_x_loader_fscore = torch.utils.data.DataLoader(train_x, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=4)
    test_x_loader_fscore = torch.utils.data.DataLoader(test_x, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=4)

    # choose feature by MIFS
    file = open(FC_mifs, 'r')
    selected_index = file.read().split(' ')
    selected_index = map(int, selected_index[: -1])
    print("numbers of selected feature for MIFS: %d" %(len(selected_index)))
    file.close()
    selected_feature = []
    for i in selected_index:
        selected_feature.append(x_columns[i])
    train_x = np.asarray(pd.DataFrame(train_data, columns=selected_feature))
    test_x = np.asarray(pd.DataFrame(test_data, columns=selected_feature))
    train_x_loader_mifs = torch.utils.data.DataLoader(train_x, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=4)
    test_x_loader_mifs = torch.utils.data.DataLoader(test_x, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=4)

    # choose feature by randomforestclassifier
    rf0 = RandomForestClassifier()
    rf0.fit(train_x, train_y)
    importances = rf0.feature_importances_

    for feature, score in zip(x_columns, importances):
        if score <= 0.0001:
            train_data = train_data.drop(feature, axis = 1)
            test_data = test_data.drop(feature, axis = 1)
    train_data = train_data.drop(['subject', 'Activity'], axis=1)
    test_data = test_data.drop(['subject', 'Activity'], axis=1)
    import_feature = len(train_data.columns)
    file1 = open("train and test data imformation.txt", "w")
    file1.write('Selected feature numbers(randomforestclassifier): %d\n\n' %(import_feature))
    file1.write('Selected feature columns(randomforestclassifier):')
    for name_feature in train_data.columns:
        file1.write('%s ' %(name_feature))
    file1.write("\n\n")
    train_x1 = np.asarray(train_data)
    test_x1 = np.asarray(test_data)
    train_import_x_loader = torch.utils.data.DataLoader(train_x1, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=4)
    test_import_x_loader = torch.utils.data.DataLoader(test_x1, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=4)

    # write count for 6 classes in train and test dataset into file
    train_total_dict = defaultdict(int)
    test_total_dict = defaultdict(int)
    for i in train_y:
        train_total_dict[i] += 1
    for i in test_y:
        test_total_dict[i] += 1
    file1.write('Numbers of 6 Activities in Train:\n')
    for key, item in train_total_dict.items():
        file1.write('%s: %d\n' %(classes[key], item))
    file1.write('Numbers of 6 Activities in Test:\n')
    for key, item in test_total_dict.items():
        file1.write('%s: %d\n' %(classes[key], item))
    file1.close()

    # build label dataloader
    train_y_loader = torch.utils.data.DataLoader(train_y, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=4)
    test_y_loader = torch.utils.data.DataLoader(test_y, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=4)
    return classes, train_total_dict, test_total_dict, import_feature, train_x_loader,\
            train_y_loader, test_x_loader, test_y_loader, train_x_loader_fscore, test_x_loader_fscore,\
            train_x_loader_mifs, test_x_loader_mifs, train_import_x_loader, test_import_x_loader, train_y, test_y

def Classifier(net, file1, file2, optimizer, BATCH_SIZE, MAX_EPOCH, classes, train_total_dict, 
    test_total_dict, train_x_loader, train_y_loader, test_x_loader, test_y_loader):
    """
    in Classifier function, we train the model and predict test label. return loss results,
    prediction results and 6 activities prediction result.
    """
    train_loss_model = []
    train_acc_model = []
    test_loss_model = []
    test_acc_model = []
    laying_acc = []
    sitting_acc = []
    standing_acc = []
    walking_acc = []
    walkingdown_acc = []
    walkingup_acc = []
    criterion = nn.CrossEntropyLoss()
    test_acc_pre = 0
    for epoch in range(MAX_EPOCH):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(zip(train_x_loader, train_y_loader)):
            inputs, labels = (Variable(inputs).float()).cuda(), Variable(labels).cuda()
            optimizer.zero_grad()
            net.batch_size = len(labels)
            net.hidden = net.init_hidden()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print('Finish training EPOCH %d, start evaluating...' %(epoch+1))
        train_loss, train_acc, train_acc_dict, train_prediction, net = eval_net(train_x_loader, train_y_loader, train_total_dict, net)
        test_loss, test_acc, test_acc_dict, test_prediction, net = eval_net(test_x_loader, test_y_loader, test_total_dict, net)
        train_loss_model.append(train_loss)
        train_acc_model.append(train_acc)
        test_loss_model.append(test_loss)
        test_acc_model.append(test_acc)
        laying_acc.append(test_acc_dict[0])
        sitting_acc.append(test_acc_dict[1])
        standing_acc.append(test_acc_dict[2])
        walking_acc.append(test_acc_dict[3])
        walkingdown_acc.append(test_acc_dict[4])
        walkingup_acc.append(test_acc_dict[5])
        # we only output the best prediction
        if test_acc_pre < test_acc:
            best_prediction = test_prediction
            test_acc_pre = test_acc
        # print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
        #       (epoch+1, train_loss, train_acc, test_loss, test_acc))
        WriteDictResult(file2, epoch, classes, test_acc_dict)
        WriteResult(file1, epoch, train_loss, train_acc, test_loss, test_acc)
    return train_loss_model, train_acc_model, test_loss_model, test_acc_model,laying_acc,\
            sitting_acc, standing_acc, walking_acc, walkingdown_acc, walkingup_acc, best_prediction

def PlotConfusionMatrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(11.2, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=10)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("%s.png" %(str(title)))

if __name__ == "__main__":

    # Pparameter define
    BATCH_SIZE = 32 # mini_batch size, can be changed
    MAX_EPOCH = 50 # maximum epoch to train, can be changed
    NUM_FEATURE = 561 # input feature numbers, cannot be changed

    # data loaders. classes is 6 activities.
    # train_total_dict and test_total_dict is count for 6 activities.
    train = "train.csv"
    test = "test.csv"
    FC_fscore = "selected feature f_score.txt" 
    FC_mifs = "selected feature MIFS.txt" 
    print("Data loading and processing...")
    classes, train_total_dict, test_total_dict, import_feature,\
    train_x_loader, train_y_loader, test_x_loader, test_y_loader, train_x_loader_fscore, test_x_loader_fscore,\
    train_x_loader_mifs, test_x_loader_mifs, train_import_x_loader,\
    test_import_x_loader, train_y, test_y = LoadData(train, test, FC_fscore, FC_mifs)
    print("Finished data loading and processing...")
 
    # MLP Classifier, train 50 epoches, return train loss, test loss, train acc, test acc,
    # and test acc for different activity. Best test acc around 95%
    net = MLPNet(num_feature=NUM_FEATURE, hidden_dim=512, batch_size=BATCH_SIZE)
    net = net.cuda()
    net.train()
    file1 = open("MLP Classifier Result.txt", "w")
    file2 = open("MLP Classifier Dict Result.txt", "w")
    file1.write('Number of select feature: %d\n' %(NUM_FEATURE))
    file2.write('Number of select feature: %d\n' %(NUM_FEATURE))
    optimizer = optim.SGD(net.parameters(), lr=0.0009, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.00043, betas=(0.9, 0.999), eps=1e-1)
    print('Start MLP training...')
    train_loss_MLP, train_acc_MLP, test_loss_MLP, test_acc_MLP,\
    laying_acc_MLP, sitting_acc_MLP, standing_acc_MLP, walking_acc_MLP, walkingdown_acc_MLP,\
    walkingup_acc_MLP, test_prediction_MLP = Classifier(net, file1, file2, optimizer, BATCH_SIZE, MAX_EPOCH, classes, train_total_dict, 
                                                        test_total_dict, train_x_loader, train_y_loader, test_x_loader, test_y_loader)
    file1.close()
    file2.close()
    print('Finished MLP Training...')

    # LSTM Classifier, train 50 epoches, return train loss, test loss, train acc, test acc,
    # and test acc for different activity. Best test acc around 95%
    net = LSTMNet(num_feature=NUM_FEATURE, hidden_dim=512, batch_size=BATCH_SIZE)
    net = net.cuda()
    net.train()
    file1 = open("LSTM Classifier Result.txt", "w")
    file2 = open("LSTM Classifier Dict Result.txt", "w")
    file1.write('Number of select feature: %d\n' %(NUM_FEATURE))
    file2.write('Number of select feature: %d\n' %(NUM_FEATURE))
    optimizer = optim.SGD(net.parameters(), lr=0.0016, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.00063, betas=(0.9, 0.999), eps=1e-2)
    print('Start LSTM training...')
    train_loss_LSTM, train_acc_LSTM, test_loss_LSTM, test_acc_LSTM,laying_acc_LSTM,\
    sitting_acc_LSTM, standing_acc_LSTM,walking_acc_LSTM,walkingdown_acc_LSTM,\
    walkingup_acc_LSTM, test_prediction_LSTM = Classifier(net, file1, file2, optimizer, BATCH_SIZE, MAX_EPOCH, classes, train_total_dict,  
                                                         test_total_dict, train_x_loader, train_y_loader, test_x_loader, test_y_loader)
    file1.close()
    file2.close()
    print('Finished LSTM Training...')


    NUM_FEATURE = import_feature
    # Importance MLP Classifier, train 50 epoches, return train loss, test loss, train acc, test acc,
    # and test acc for different activity. Best test acc around 94% - 95.5%
    net = MLPNet(num_feature=NUM_FEATURE, hidden_dim=512, batch_size=BATCH_SIZE)
    net = net.cuda()
    net.train()
    file1 = open("MLP Classifier Result(fc).txt", "w")
    file2 = open("MLP Classifier Dict Result(fc).txt", "w")
    file1.write('Number of select feature: %d\n' %(NUM_FEATURE))
    file2.write('Number of select feature: %d\n' %(NUM_FEATURE))
    optimizer = optim.SGD(net.parameters(), lr=0.0013, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.0008, betas=(0.9, 0.999))
    print('Start Importance Feature Selection MLP training...')
    train_loss_MLP_fc, train_acc_MLP_fc, test_loss_MLP_fc, test_acc_MLP_fc,\
    laying_acc_MLP_fc, sitting_acc_MLP_fc, standing_acc_MLP_fc, walking_acc_MLP_fc, walkingdown_acc_MLP_fc,\
    walkingup_acc_MLP_fc, test_prediction_MLP_fc = Classifier(net, file1, file2, optimizer, BATCH_SIZE, MAX_EPOCH, classes, train_total_dict, 
                                                        test_total_dict, train_import_x_loader, train_y_loader, test_import_x_loader, test_y_loader)
    file1.close()
    file2.close()
    print('Finished Importance Feature Selection MLP Training...')

    # Importance LSTM Classifier, train 50 epoches, return train loss, test loss, train acc, test acc,
    # and test acc for different activity. Best test acc around 94% - 95.5%.
    net = LSTMNet(num_feature=NUM_FEATURE, hidden_dim=512, batch_size=BATCH_SIZE)
    net = net.cuda()
    net.train()
    file1 = open("LSTM Classifier Result(fc).txt", "w")
    file2 = open("LSTM Classifier Dict Result(fc).txt", "w")
    file1.write('Number of select feature: %d\n' %(NUM_FEATURE))
    file2.write('Number of select feature: %d\n' %(NUM_FEATURE))
    optimizer = optim.SGD(net.parameters(), lr=0.0013, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-6)
    print('Start Importance Feature Selection LSTM training...')
    train_loss_LSTM_fc, train_acc_LSTM_fc, test_loss_LSTM_fc, test_acc_LSTM_fc,laying_acc_LSTM_fc,\
    sitting_acc_LSTM_fc, standing_acc_LSTM_fc, walking_acc_LSTM_fc, walkingdown_acc_LSTM_fc,\
    walkingup_acc_LSTM_fc, test_prediction_LSTM_fc = Classifier(net, file1, file2, optimizer, BATCH_SIZE, MAX_EPOCH, classes, train_total_dict,  
                                                         test_total_dict, train_import_x_loader, train_y_loader, test_import_x_loader, test_y_loader)
    file1.close()
    file2.close()
    print('Finished Importance Feature Selection LSTM Training...')

    NUM_FEATURE = 400
    # MIFS MLP Classifier, 400 features. train 50 epoches, return train loss, test loss, train acc, test acc,
    # and test acc for different activity. Best test acc around 91% - 95%.
    net = MLPNet(num_feature=NUM_FEATURE, hidden_dim=512, batch_size=BATCH_SIZE)
    net = net.cuda()
    net.train()
    file1 = open("MLP Classifier Result(mifs).txt", "w")
    file2 = open("MLP Classifier Dict Result(mifs).txt", "w")
    file1.write('Number of select feature: %d\n' %(NUM_FEATURE))
    file2.write('Number of select feature: %d\n' %(NUM_FEATURE))
    optimizer = optim.SGD(net.parameters(), lr=0.0015, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-1)
    print('Start MIFS MLP training...')
    train_loss_MLP_mifs, train_acc_MLP_mifs, test_loss_MLP_mifs, test_acc_MLP_mifs,\
    laying_acc_MLP_mifs, sitting_acc_MLP_mifs, standing_acc_MLP_mifs, walking_acc_MLP_mifs, walkingdown_acc_MLP_mifs,\
    walkingup_acc_MLP_mifs, test_prediction_MLP_mifs = Classifier(net, file1, file2, optimizer, BATCH_SIZE, MAX_EPOCH, classes, train_total_dict, 
                                                        test_total_dict, train_x_loader_mifs, train_y_loader, test_x_loader_mifs, test_y_loader)
    file1.close()
    file2.close()
    print('Finished MIFS MLP Training...')

    # MIFS LSTM Classifier, 400 features. train 50 epoches, return train loss, test loss, train acc, test acc,
    # and test acc for different activity. Best test acc around 91% - 95%.
    net = LSTMNet(num_feature=NUM_FEATURE, hidden_dim=512, batch_size=BATCH_SIZE)
    net = net.cuda()
    net.train()
    file1 = open("LSTM Classifier Result(mifs).txt", "w")
    file2 = open("LSTM Classifier Dict Result(mifs).txt", "w")
    file1.write('Number of select feature: %d\n' %(NUM_FEATURE))
    file2.write('Number of select feature: %d\n' %(NUM_FEATURE))
    optimizer = optim.SGD(net.parameters(), lr=0.0015, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.00075, betas=(0.9, 0.999), eps=1e-1)
    print('Start MIFS LSTM training...')
    train_loss_LSTM_mifs, train_acc_LSTM_mifs, test_loss_LSTM_mifs, test_acc_LSTM_mifs, laying_acc_LSTM_mifs,\
    sitting_acc_LSTM_mifs, standing_acc_LSTM_mifs, walking_acc_LSTM_mifs, walkingdown_acc_LSTM_mifs,\
    walkingup_acc_LSTM_mifs, test_prediction_LSTM_mifs = Classifier(net, file1, file2, optimizer, BATCH_SIZE, MAX_EPOCH, classes, train_total_dict,  
                                                         test_total_dict, train_x_loader_mifs, train_y_loader, test_x_loader_mifs, test_y_loader)
    file1.close()
    file2.close()
    print('Finished MIFS LSTM Training...')

    # F-score MLP Classifier, 400 features. train 50 epoches, return train loss, test loss, train acc, test acc,
    # and test acc for different activity. Best test acc around 90% - 94%.
    net = MLPNet(num_feature=NUM_FEATURE, hidden_dim=512, batch_size=BATCH_SIZE)
    net = net.cuda()
    net.train()
    file1 = open("MLP Classifier Result(fscore).txt", "w")
    file2 = open("MLP Classifier Dict Result(fscore).txt", "w")
    file1.write('Number of select feature: %d\n' %(NUM_FEATURE))
    file2.write('Number of select feature: %d\n' %(NUM_FEATURE))
    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-1)
    print('Start F-score MLP training...')
    train_loss_MLP_fscore, train_acc_MLP_fscore, test_loss_MLP_fscore, test_acc_MLP_fscore,\
    laying_acc_MLP_fscore, sitting_acc_MLP_fscore, standing_acc_MLP_fscore, walking_acc_MLP_fscore, walkingdown_acc_MLP_fscore,\
    walkingup_acc_MLP_fscore, test_prediction_MLP_fscore = Classifier(net, file1, file2, optimizer, BATCH_SIZE, MAX_EPOCH, classes, train_total_dict, 
                                                        test_total_dict, train_x_loader_fscore, train_y_loader, test_x_loader_fscore, test_y_loader)
    file1.close()
    file2.close()
    print('Finished F-score MLP Training...')

    # MIFS LSTM Classifier, 400 features. train 50 epoches, return train loss, test loss, train acc, test acc,
    # and test acc for different activity. Best test acc around 91% - 95%.
    net = LSTMNet(num_feature=NUM_FEATURE, hidden_dim=512, batch_size=BATCH_SIZE)
    net = net.cuda()
    net.train()
    file1 = open("LSTM Classifier Result(fscore).txt", "w")
    file2 = open("LSTM Classifier Dict Result(fscore).txt", "w")
    file1.write('Number of select feature: %d\n' %(NUM_FEATURE))
    file2.write('Number of select feature: %d\n' %(NUM_FEATURE))
    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.00075, betas=(0.9, 0.999), eps=1e-1)
    print('Start F-score LSTM training...')
    train_loss_LSTM_fscore, train_acc_LSTM_fscore, test_loss_LSTM_fscore, test_acc_LSTM_fscore, laying_acc_LSTM_fscore,\
    sitting_acc_LSTM_fscore, standing_acc_LSTM_fscore, walking_acc_LSTM_fscore, walkingdown_acc_LSTM_fscore,\
    walkingup_acc_LSTM_fscore, test_prediction_LSTM_fscore = Classifier(net, file1, file2, optimizer, BATCH_SIZE, MAX_EPOCH, classes, train_total_dict,  
                                                         test_total_dict, train_x_loader_fscore, train_y_loader, test_x_loader_fscore, test_y_loader)
    file1.close()
    file2.close()
    print('Finished F-score LSTM Training...')

    # plot figure
    print('Start Ploting...')
    file = open("index for best accuracy.txt", 'w')
    file.write("index selected in MLP and LSTM:\n")
    figurePlot(MAX_EPOCH, classes, train_loss_MLP, test_loss_MLP, train_loss_LSTM,
                test_loss_LSTM, train_acc_MLP, test_acc_MLP, train_acc_LSTM, test_acc_LSTM,
                laying_acc_MLP, sitting_acc_MLP, standing_acc_MLP, walking_acc_MLP,
                walkingdown_acc_MLP, walkingup_acc_MLP, laying_acc_LSTM, sitting_acc_LSTM,
                standing_acc_LSTM, walking_acc_LSTM, walkingdown_acc_LSTM, walkingup_acc_LSTM, file,
                ['Loss', 'Accuracy'], ['Train MLP', 'Test MLP', 'Train LSTM', 'Test LSTM'], ['MLP', 'LSTM'])
    file.write("index selected in MLP and importance MLP:\n")
    figurePlot(MAX_EPOCH, classes, train_loss_MLP, test_loss_MLP, train_loss_MLP_fc,
                test_loss_MLP_fc, train_acc_MLP, test_acc_MLP, train_acc_MLP_fc, test_acc_MLP_fc,
                laying_acc_MLP, sitting_acc_MLP, standing_acc_MLP, walking_acc_MLP,
                walkingdown_acc_MLP, walkingup_acc_MLP, laying_acc_MLP_fc, sitting_acc_MLP_fc,
                standing_acc_MLP_fc, walking_acc_MLP_fc, walkingdown_acc_MLP_fc, walkingup_acc_MLP_fc, file,
                ['Loss', 'Accuracy'], ['Train MLP', 'Test MLP', 'Importance Train MLP', 'Importance Test MLP'], ['MLP', 'Importance MLP'])
    file.write("index selected in LSTM and importance LSTM:\n")
    figurePlot(MAX_EPOCH, classes, train_loss_LSTM, test_loss_LSTM, train_loss_LSTM_fc,
                test_loss_LSTM_fc, train_acc_LSTM, test_acc_LSTM, train_acc_LSTM_fc, test_acc_LSTM_fc,
                laying_acc_LSTM, sitting_acc_LSTM, standing_acc_LSTM, walking_acc_LSTM,
                walkingdown_acc_LSTM, walkingup_acc_LSTM, laying_acc_LSTM_fc, sitting_acc_LSTM_fc,
                standing_acc_LSTM_fc, walking_acc_LSTM_fc, walkingdown_acc_LSTM_fc, walkingup_acc_LSTM_fc, file,
                ['Loss', 'Accuracy'], ['Train LSTM', 'Test LSTM', 'Importance Train LSTM', 'Importance Test LSTM'], ['LSTM', 'Importance LSTM'])
    figurePlot(MAX_EPOCH, classes, train_loss_MLP_fc, test_loss_MLP_fc, train_loss_LSTM_fc,
                test_loss_LSTM_fc, train_acc_MLP_fc, test_acc_MLP_fc, train_acc_LSTM_fc, test_acc_LSTM_fc,
                laying_acc_MLP_fc, sitting_acc_MLP_fc, standing_acc_MLP_fc, walking_acc_MLP_fc,
                walkingdown_acc_MLP_fc, walkingup_acc_MLP_fc, laying_acc_LSTM_fc, sitting_acc_LSTM_fc,
                standing_acc_LSTM_fc, walking_acc_LSTM_fc, walkingdown_acc_LSTM_fc, walkingup_acc_LSTM_fc, file,
                ['Loss', 'Accuracy'], ['Importance Train MLP', 'Importance Test MLP', 'Importance Train LSTM', 'Importance Test LSTM'],
                 ['Importance MLP', 'Importance LSTM'])
    file.write("index selected in MLP and MIFS MLP:\n")
    figurePlot(MAX_EPOCH, classes, train_loss_MLP, test_loss_MLP, train_loss_MLP_mifs,
                test_loss_MLP_mifs, train_acc_MLP, test_acc_MLP, train_acc_MLP_mifs, test_acc_MLP_mifs,
                laying_acc_MLP, sitting_acc_MLP, standing_acc_MLP, walking_acc_MLP,
                walkingdown_acc_MLP, walkingup_acc_MLP, laying_acc_MLP_mifs, sitting_acc_MLP_mifs,
                standing_acc_MLP_mifs, walking_acc_MLP_mifs, walkingdown_acc_MLP_mifs, walkingup_acc_MLP_mifs, file,
                ['Loss', 'Accuracy'], ['Train MLP', 'Test MLP', 'MIFS Train MLP', 'MIFS Test MLP'], ['MLP', 'MIFS MLP'])
    file.write("index selected in LSTM and MIFS LSTM:\n")
    figurePlot(MAX_EPOCH, classes, train_loss_LSTM, test_loss_LSTM, train_loss_LSTM_mifs,
                test_loss_LSTM_mifs, train_acc_LSTM, test_acc_LSTM, train_acc_LSTM_mifs, test_acc_LSTM_mifs,
                laying_acc_LSTM, sitting_acc_LSTM, standing_acc_LSTM, walking_acc_LSTM,
                walkingdown_acc_LSTM, walkingup_acc_LSTM, laying_acc_LSTM_mifs, sitting_acc_LSTM_mifs,
                standing_acc_LSTM_mifs, walking_acc_LSTM_mifs, walkingdown_acc_LSTM_mifs, walkingup_acc_LSTM_mifs, file,
                ['Loss', 'Accuracy'], ['Train LSTM', 'Test LSTM', 'MIFS Train LSTM', 'MIFS Test LSTM'], ['LSTM', 'MIFS LSTM'])
    file.write("index selected in MLP and F-score MLP:\n")
    figurePlot(MAX_EPOCH, classes, train_loss_MLP, test_loss_MLP, train_loss_MLP_fscore,
                test_loss_MLP_fscore, train_acc_MLP, test_acc_MLP, train_acc_MLP_fscore, test_acc_MLP_fscore,
                laying_acc_MLP, sitting_acc_MLP, standing_acc_MLP, walking_acc_MLP,
                walkingdown_acc_MLP, walkingup_acc_MLP, laying_acc_MLP_fscore, sitting_acc_MLP_fscore,
                standing_acc_MLP_fscore, walking_acc_MLP_fscore, walkingdown_acc_MLP_fscore, walkingup_acc_MLP_fscore, file,
                ['Loss', 'Accuracy'], ['Train MLP', 'Test MLP', 'F-score Train MLP', 'F-score Test MLP'], ['MLP', 'F-score MLP'])
    file.write("index selected in LSTM and F-score LSTM\n")
    figurePlot(MAX_EPOCH, classes, train_loss_LSTM, test_loss_LSTM, train_loss_LSTM_fscore,
                test_loss_LSTM_fscore, train_acc_LSTM, test_acc_LSTM, train_acc_LSTM_fscore, test_acc_LSTM_fscore,
                laying_acc_LSTM, sitting_acc_LSTM, standing_acc_LSTM, walking_acc_LSTM,
                walkingdown_acc_LSTM, walkingup_acc_LSTM, laying_acc_LSTM_fscore, sitting_acc_LSTM_fscore,
                standing_acc_LSTM_fscore, walking_acc_LSTM_fscore, walkingdown_acc_LSTM_fscore, walkingup_acc_LSTM_fscore, file,
                ['Loss', 'Accuracy'], ['Train LSTM', 'Test LSTM', 'F-score Train LSTM', 'F-score Test LSTM'], ['LSTM', 'F-score LSTM'])
    file.close()
    print('Finished Ploting...')

    # plot confusion matrix
    print('Start confusion_matrix...')
    cfs1 = confusion_matrix(test_y, np.asarray(test_prediction_MLP))
    cfs2 = confusion_matrix(test_y, np.asarray(test_prediction_LSTM))
    cfs3 = confusion_matrix(test_y, np.asarray(test_prediction_MLP_fc))
    cfs4 = confusion_matrix(test_y, np.asarray(test_prediction_LSTM_fc))
    cfs5 = confusion_matrix(test_y, np.asarray(test_prediction_MLP_mifs))
    cfs6 = confusion_matrix(test_y, np.asarray(test_prediction_LSTM_mifs))
    cfs7 = confusion_matrix(test_y, np.asarray(test_prediction_MLP_fscore))
    cfs8 = confusion_matrix(test_y, np.asarray(test_prediction_LSTM_fscore))
    PlotConfusionMatrix(cfs1, classes = classes, title = "MLP Confusion Matrix without Feature Selection")
    PlotConfusionMatrix(cfs2, classes = classes, title = "LSTM Confusion Matrix Without Feature Selection")
    PlotConfusionMatrix(cfs3, classes = classes, title = "MLP Confusion Matrix With Importance Feature Selection")
    PlotConfusionMatrix(cfs4, classes = classes, title = "LSTM Confusion Matrix With Importance Feature Selection")
    PlotConfusionMatrix(cfs5, classes = classes, title = "MLP Confusion Matrix With MIFS Feature Selection")
    PlotConfusionMatrix(cfs6, classes = classes, title = "LSTM Confusion Matrix With MIFS Feature Selection")
    PlotConfusionMatrix(cfs7, classes = classes, title = "MLP Confusion Matrix With F-score Feature Selection")
    PlotConfusionMatrix(cfs8, classes = classes, title = "LSTM Confusion Matrix With F-score Feature Selection")
    print('Finished confusion_matrix...')















