import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from Blur.BlurCNN import BlurCNN
from CustomMetrics.CustomMetrics import CustomAccurencyMetrics

# "Blur" CNN manager class
# Input: Focused image
# Output: Blured image
# Idea: Train CNN to contain blur operator and make PSF from one-pixel image
#       Also can be used for generate syntetic data with PSF from microscope to "Deblur" CNN
class BlurTrainer():
    # constructor
    def __init__(self):
        self.model = BlurCNN()
        return

    # Method which provides loading model state from file
    def LoadModel(self, modelPath):
        self.model.load_state_dict(torch.load(modelPath))
        return

    # Method witch provide datasets
    def SetDatasets(self, trainDS, valDS):
        self.train_set = trainDS
        self.test_set = valDS
        return

    # Method which provides loading model state from file
    def Train(self, epochs=15, learning_rate=0.001, isNeedToPlotLossGraph=True):
        # init optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        loss_func = nn.L1Loss()

        # init must-have data for losses and accurency
        train_average_losses = list()
        test_average_losses = list()
        acc_list = list()
        testConvolver = CustomAccurencyMetrics()
        
        # train model loop
        for epoch in range(epochs):
            # Firstly, train on train_set
            current_train_loss = 0
            for i, (focused, tru_img) in enumerate(self.train_set):
                pred_img = self.model(focused)
            
                # count loss
                loss = loss_func(tru_img, pred_img)
                current_train_loss = current_train_loss + loss.item()

                # optimize loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # count accuracy
                accuracy = testConvolver.CountAccuracy(pred_img, tru_img, eps = 0.005)

                # detach for memory
                loss = loss.detach()
                pred_img = pred_img.detach()

                print('Epoch [{}/{}], Loss: {:.6f}, accuracy: {:.3f}'.format(epoch + 1, epochs, loss.item(), accuracy))

            train_average_losses.append(current_train_loss / len(self.train_set))

            # Secondly, check losses on validate data set
            current_val_loss = 0
            for i, (focused, tru_img) in enumerate(self.test_set):
                pred_img = self.model(focused)

                loss = loss_func(tru_img, pred_img)
                current_val_loss = current_val_loss + loss.item()

                loss = loss.detach()
                pred_img = pred_img.detach()
                
            test_average_losses.append(current_val_loss / len(self.test_set))

        # plot losses info
        if isNeedToPlotLossGraph:
            x = np.arange(1, epochs + 1, 1)
            plt.title('Losses graph')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.plot(x, train_average_losses, 'r-', x, test_average_losses, 'b-')
            plt.grid(True)
            plt.show()
        return

    # Method wich provides result of CNN from some data
    def Run(self, data):
        return self.model(data)

    def SaveModel(self, path):
        torch.save(self.model.state_dict(), path)
        return
