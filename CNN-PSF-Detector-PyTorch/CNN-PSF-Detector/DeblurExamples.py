'''This file includes ONLY examples with using that "Deblur" CNN model'''

from Deblur.DeblurDataGenerator import DeblurDataGenerator
from Deblur.DeblurTrainer import DeblurTrainer

def deblur_run():
    # generate data set
    generator = DeblurDataGenerator()
    
    # Init model trainer params
    big_epochs = 10
    dif_epochs = 4
    lil_epochs = 4
    learning_rate = 0.001
    deblurTrainer = DeblurTrainer()
    
    # Init Data-sets
    big_train_set = generator.DataSetGenerator(dataSetPower=100, batch_size=1, rad_range=(11, 13), intensity_range=(64, 128), padding_range=(-5, 5))
    big_test_set = generator.DataSetGenerator(dataSetPower=25, batch_size=1, rad_range=(11, 13), intensity_range=(64, 128), padding_range=(-5, 5))

    dif_train_set = generator.DataSetGenerator(dataSetPower=400, batch_size=1, rad_range=(3, 12), intensity_range=(32, 128), padding_range=(-5, 5))
    dif_test_set = generator.DataSetGenerator(dataSetPower=100, batch_size=1, rad_range=(3, 12), intensity_range=(32, 128), padding_range=(-5, 5))

    lil_train_set = generator.DataSetGenerator(dataSetPower=100, batch_size=1, rad_range=(1, 6), intensity_range=(16, 96), padding_range=(-3, 3))
    lil_test_set = generator.DataSetGenerator(dataSetPower=25, batch_size=1, rad_range=(1, 6), intensity_range=(16, 96), padding_range=(-3, 3))

    # train model
    deblurTrainer.SetDatasets(big_train_set, big_test_set)
    deblurTrainer.Train(epochs=big_epochs, learning_rate=learning_rate, isNeedToPlotLossGraph=True)
    
    # train model
    deblurTrainer.SetDatasets(dif_train_set, dif_test_set)
    deblurTrainer.Train(epochs=dif_epochs, learning_rate=learning_rate, isNeedToPlotLossGraph=True)
    
    # train model
    deblurTrainer.SetDatasets(lil_train_set, lil_test_set)
    deblurTrainer.Train(epochs=lil_epochs, learning_rate=learning_rate, isNeedToPlotLossGraph=True)

    # save model
    blurTrainer.SaveModel('models/deblur_test_model_1.cnmd')
    return
