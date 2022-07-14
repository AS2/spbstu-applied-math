import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Classes witch provides taking custom losses
def CountBluredPoint(bluredImage, circle, psf, row, col, batch_elem, border = 1.0):
    psfTmp = psf.mul(circle[batch_elem][0][row][col])

    circleX = 36#circle.shape[0]
    circleY = 36#circle.shape[1]

    psfX = 31#psf.shape[0]
    psfY = 31#psf.shape[1]

    bI_shiftedRow = row - int(psfX / 2)
    bI_shiftedCol = col - int(psfY / 2)

    for row_psf in range(psfX):
        if bI_shiftedRow + row_psf >= 0 and bI_shiftedRow + row_psf < circleX:
            for col_psf in range(psfY):
                if bI_shiftedCol + col_psf >= 0 and bI_shiftedCol + col_psf < circleY:
                    x = bI_shiftedRow + row_psf
                    y = bI_shiftedCol + col_psf
                    bluredImage[batch_elem][0][x][y] = min(bluredImage[batch_elem][0][x][y] + psfTmp[batch_elem][0][row_psf][col_psf], border)

    return bluredImage

def ImageConvolution2D(circle, psf):
    blured = torch.zeros(circle.size())
    blured = blured.cuda()

    for batch_elem in range(len(circle)):
        for row in range(len(circle[0][0])):
            for col in range(len(circle[0][0][0])):
                if (circle[batch_elem][0][row][col] != 0):
                    #print("Batch: {}, row: {}, col: {}".format(batch_elem, row, col))
                    blured = CountBluredPoint(blured, circle, psf, row, col, batch_elem)
    return blured

# Class witch using barriers for taking output values into [0; 1]
def PSFTestLoss(x_arg, psf_predicted, y_true):
    # count predicted y
    y_pred = ImageConvolution2D(y_true, psf_predicted)
    return torch.mean(torch.square(y_pred - x_arg))

def ImageAdvancedConvolution2D(circle, psf, border = 1.):
    bluredImage = torch.zeros(circle.size())

    batches = len(circle)

    psfSize = len(psf[0][0])
    psfRad = int(psfSize / 2)

    circleSize = len(circle[0][0])

    for batch in range(batches):
        for row in range(circleSize):
            for col in range(circleSize):
                point = 0
                rowStart = row - psfRad
                colStart = col - psfRad

                for psf_row in range(psfSize):
                    if rowStart + psf_row >= 0 and rowStart  + psf_row < circleSize:
                        for psf_col in range(psfSize):
                            if colStart + psf_col >= 0 and colStart  + psf_col < circleSize:
                                point = min(point + circle[batch][0][rowStart + psf_row][colStart + psf_col] * psf[batch][0][psfSize - 1 - psf_row][psfSize - 1 - psf_col], border)
                            if point == border:
                                break
                    if point == border:
                        break

                bluredImage[batch][0][row][col] = point

    return bluredImage

# Class witch using barriers for taking output values into [0; 1]
def PSFTestLossAdvanced(x_arg, psf_predicted, y_true):
    # count predicted y
    y_pred = ImageAdvancedConvolution2D(y_true, psf_predicted)
    return torch.mean(torch.square(y_pred - x_arg))

class PSFTrainController():
    def __init__(self):
        return

    def castImageToTorch(self, image):
        image_torch = torch.from_numpy(image.reshape(1, 1, image.shape[0], image.shape[0]))
        return image_torch

    def doubleTransposePSF(self, psf):
        rows = int(len(psf) / 2) + 1

        for row in range(rows):
            cols = len(psf) if (rows != row + 1) else rows
            for col in range(cols):
                    tmp = psf[row][col]
                    psf[row][col] = psf[len(psf) - 1 - row][len(psf) - 1 - col]
                    psf[len(psf) - 1 - row][len(psf) - 1 - col] = tmp
        return psf

    def testConvOnImages(self, label, psf):
        psf_torch = self.castImageToTorch(self.doubleTransposePSF(psf))
        label_torch = self.castImageToTorch(label)
        return F.conv2d(label_torch, psf_torch, padding='same')

    def testConvOnSingleBatchTorch(self, label, psf):
        # TODO: переворачивать надо по идее....
        return F.conv2d(label, psf, padding='same')

    def CountAccuracy(self, pred_image, true_image, eps = 0.001):
        accuracy = 0
        total_pxls = 36 * 36

        for x in range(36):
            for y in range(36):
                accuracy = accuracy + (abs(pred_image[0][0][x][y] - true_image[0][0][x][y]) < eps)
        return float(accuracy / total_pxls) * 100
