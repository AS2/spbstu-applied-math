from PIL import Image
import numpy as np
import torch

from math import sqrt
from random import randint

# Class which provides generating syntetic data for "Deblur" CNN
# Output: array of pairs: (1x1x36x36, 1x1x36x36) = [blured_image, focused_image]
class DeblurDataGenerator:
    # constructor
    def __init__(self):
        return

    def CircleGenerator(self, shape=(36, 36), centerPadding = (0, 0), pixelRadius = 6, intensity = 100):
        circle = np.zeros(shape)
        centerCoords = [shape[0] / 2 - centerPadding[0], shape[1] / 2 - centerPadding[1]]

        for x in range(len(circle)):
            for y in range(len(circle[0])):
                if (x - centerCoords[0]) ** 2 + (y - centerCoords[1]) ** 2 <= pixelRadius * pixelRadius:
                    circle[x][y] = (128 + intensity * (pixelRadius * pixelRadius - ((x - centerCoords[0]) ** 2 + (y - centerCoords[1]) ** 2)) / (pixelRadius * pixelRadius))
        return circle

    def EmulatePSF(self, rad = 9., shape=(31, 31)):
        emulatedPSF = np.zeros(shape)

        centerX, centerY = int(shape[0] / 2), int(shape[1] / 2)

        # make unfocused circle
        for row in range(shape[0]):
            for col in range(shape[1]):
                if (row - centerX) ** 2 + (col - centerY) ** 2 <= rad * rad:
                    emulatedPSF[row][col] = 255 * (((rad * rad - ((row - centerX) ** 2 + (col - centerY) ** 2)) / (rad * rad)) ** 5)

        # make some shifts in rows
        for row in range(shape[0]):
            shiftsCnt = int(sqrt(abs(row - centerX)))
            for pxl in range(shape[1]):
                if (pxl + shiftsCnt) < shape[1]:
                    emulatedPSF[row][pxl] = emulatedPSF[row][pxl + shiftsCnt]
                else:
                    emulatedPSF[row][pxl] = 0

        # count psf pxls sum
        normalConst = 0
        for x in range(shape[0]):
            for y in range(shape[1]):
                normalConst = normalConst + emulatedPSF[x][y]

        # and normalize
        return emulatedPSF / normalConst

    def CountBluredPoint(self, bluredImage, circle, psf, row, col, border = 255):
        psfTmp = psf * circle[row][col]

        bI_shiftedRow = row - int(psf.shape[0] / 2)
        bI_shiftedCol = col - int(psf.shape[0] / 2)

        for row_psf in range(psfTmp.shape[0]):
            if bI_shiftedRow + row_psf >= 0 and bI_shiftedRow + row_psf < circle.shape[0]:
                for col_psf in range(psfTmp.shape[1]):
                    if bI_shiftedCol + col_psf >= 0 and bI_shiftedCol + col_psf < circle.shape[1]:
                        bluredImage[bI_shiftedRow + row_psf][bI_shiftedCol + col_psf] = bluredImage[bI_shiftedRow + row_psf][bI_shiftedCol + col_psf] + psfTmp[row_psf][col_psf]
                        if bluredImage[bI_shiftedRow + row_psf][bI_shiftedCol + col_psf] >= border:
                            bluredImage[bI_shiftedRow + row_psf][bI_shiftedCol + col_psf] = border

        return bluredImage

    def ImageConvolution2D(self, circle, psf):
        bluredImage = np.zeros(circle.shape, dtype=float)

        for row in range(len(circle)):
            for col in range(len(circle[0])):
                if (circle[row][col] != 0):
                    bluredImage = self.CountBluredPoint(bluredImage, circle, psf, row, col)
        return bluredImage

    def DataSetGenerator(self, dataSetPower = 100, batch_size = 10, rad_range=(12, 13), intensity_range=(64, 128), padding_range=(-5, 5)):
        dataSet = list()
        psf = self.EmulatePSF(shape=(35, 35))

        batches_cnt = int(dataSetPower / batch_size)

        for i in range(batches_cnt):
            newBatchCircles = list()
            newBatchBlured = list()
            print("Batch[{}/{}]".format(i+1, batches_cnt))

            for j in range(batch_size):
                centerPaddingX = randint(padding_range[0], padding_range[1])
                centerPaddingY = randint(padding_range[0], padding_range[1])
                intensity = randint(intensity_range[0], intensity_range[1])
                radius = randint(rad_range[0], rad_range[1])

                circle = self.CircleGenerator(centerPadding=(centerPaddingX, centerPaddingY), intensity=intensity, pixelRadius=radius)
                bluredCircle = self.ImageConvolution2D(circle, psf)

                dataSet.append((torch.from_numpy(np.array(circle).reshape(1, 1, 36, 36) / 255).float(), torch.from_numpy(np.array(bluredCircle).reshape(1, 1, 36, 36) / 255).float()))

        return np.array(dataSet)

    def GenerateOnePixelImage(self, shape = (36, 36)):
        centerX = int(shape[0] / 2)
        centerY = int(shape[1] / 2)

        one_pxl_img = np.zeros(shape)
        one_pxl_img[centerX][centerY] = 255
        return one_pxl_img

    def ImageTiffSaver(self, imageToSave, fileName, multiplyer = 1.):
        im = Image.fromarray(imageToSave * multiplyer)
        im.save(fileName)
        return

    def SavePSFTensor(self, psf, filename, rows = 31, cols = 31):
        psf_nparray = psf.detach().numpy()
        psf_nparray = psf_nparray.reshape(rows, cols)

        maxValue = 0
        for x in range(rows):
            for y in range(cols):
                if (psf_nparray[x][y] > maxValue):
                    maxValue = psf_nparray[x][y]
        if maxValue != 0:
            psf_nparray = (psf_nparray * (255 / maxValue)).astype('int32')
        self.ImageTiffSaver(psf_nparray, filename)
        return
