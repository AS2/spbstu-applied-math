'''This file includes ONLY examples with using that "Blur" CNN model'''
import numpy as np
import torch
import torch.nn as nn

from Blur.BlurDataGenerator import BlurDataGenerator
from Blur.BlurTrainer import BlurTrainer

# This trainig session has three parts:
# 1) Training on big circles: to prevert training CNN to give us "black" image
# 2) Fine-tuning on different circles: to make CNN more flexible to given data
# 3) Fine-tuning on little circles: to make CNN more correct to little circles (like in given task)

def blur_run():
    # generate data set
    generator = BlurDataGenerator()
    
    # Init model trainer params
    big_epochs = 10
    dif_epochs = 4
    lil_epochs = 4
    learning_rate = 0.001
    blurTrainer = BlurTrainer()
    
    # Init Data-sets
    big_train_set = generator.DataSetGenerator(dataSetPower=100, batch_size=1, rad_range=(11, 13), intensity_range=(64, 128), padding_range=(-5, 5))
    big_test_set = generator.DataSetGenerator(dataSetPower=25, batch_size=1, rad_range=(11, 13), intensity_range=(64, 128), padding_range=(-5, 5))

    dif_train_set = generator.DataSetGenerator(dataSetPower=400, batch_size=1, rad_range=(3, 12), intensity_range=(32, 128), padding_range=(-5, 5))
    dif_test_set = generator.DataSetGenerator(dataSetPower=100, batch_size=1, rad_range=(3, 12), intensity_range=(32, 128), padding_range=(-5, 5))

    lil_train_set = generator.DataSetGenerator(dataSetPower=100, batch_size=1, rad_range=(1, 6), intensity_range=(16, 96), padding_range=(-3, 3))
    lil_test_set = generator.DataSetGenerator(dataSetPower=25, batch_size=1, rad_range=(1, 6), intensity_range=(16, 96), padding_range=(-3, 3))

    # train model
    blurTrainer.SetDatasets(big_train_set, big_test_set)
    blurTrainer.Train(epochs=big_epochs, learning_rate=learning_rate, isNeedToPlotLossGraph=True)
    
    # train model
    blurTrainer.SetDatasets(dif_train_set, dif_test_set)
    blurTrainer.Train(epochs=dif_epochs, learning_rate=learning_rate, isNeedToPlotLossGraph=True)
    
    # train model
    blurTrainer.SetDatasets(lil_train_set, lil_test_set)
    blurTrainer.Train(epochs=lil_epochs, learning_rate=learning_rate, isNeedToPlotLossGraph=True)

    # save model
    blurTrainer.SaveModel('models/blur_test_model_1.cnmd')

    # generate one-pixel image and compare PSFs
    one_pixel_image = torch.from_numpy(np.array(generator.GenerateOnePixelImage() / 255.).reshape(1, 1, 36, 36)).float()
    emulatedInDataPSF = torch.from_numpy(generator.EmulatePSF().reshape(1, 1, 31, 31)).float()
    outputedPSF = blurTrainer.Run(one_pixel_image)

    generator.SavePSFTensor(emulatedInDataPSF, "emulated_psf.tiff", rows=31, cols=31)
    generator.SavePSFTensor(outputedPSF, "outputed_psf.tiff", rows=36, cols=36)
    return

def blur_model_testing(model_path='models/blur_test_model_1.cnmd'):
    blurTrainer = BlurTrainer()
    blurTrainer.LoadModel(model_path)
    generator = BlurDataGenerator()
    
    # generate one-pixel image and compare PSFs
    one_pixel_image = torch.from_numpy(np.array(generator.GenerateOnePixelImage() / 255.).reshape(1, 1, 36, 36)).float()
    emulatedInDataPSF = torch.from_numpy(generator.EmulatePSF().reshape(1, 1, 31, 31)).float()
    outputedPSF = blurTrainer.Run(one_pixel_image)

    generator.SavePSFTensor(emulatedInDataPSF, "emulated_psf.tiff", rows=31, cols=31)
    generator.SavePSFTensor(outputedPSF, "outputed_psf.tiff", rows=36, cols=36)
    
    
    #data = generator.DataSetGenerator(dataSetPower=1, batch_size=1, rad_range=(5, 7), intensity_range=(64, 128), padding_range=(-2, 2))[0]
    #outputed = blurTrainer.Run(data[0])

    #generator.SavePSFTensor(data[1], "sm_img_generated.tiff", rows=36, cols=36)
    #generator.SavePSFTensor(outputed, "sm_img_outputed.tiff", rows=36, cols=36)
    return
