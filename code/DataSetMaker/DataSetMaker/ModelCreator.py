import os
import numpy as np
from PIL import Image

from ModelConfig import ModelConfig


class ModelCreator:
    #class CreatorErrors:
    #    def __init__(self):
    #        self.CREATOR_ERRORS = {0 : ""}
    #        return

    def __init__(self, modelConfig : ModelConfig):
        self.modelConfig = modelConfig
        return

    # Функция обхода директории на поиск всех *.tiff или *.tif файлов
    def FindAllPaths(self, directoryPath : str):
        paths = list()
        for file in os.listdir(directoryPath):
            if file.endswith(".tiff") or file.endswith(".tif"):
                paths.append(directoryPath + "/" + file)
        return paths

    # Функция корректности параметров *tiff изображения
    def isTiffCorrect(self, width, height, nlayers):
        return self.modelConfig.width == width and self.modelConfig.height == height and self.modelConfig.layers == nlayers

    # Функция загрузки всех *.tiff или *.tif файлов из списка путей
    def LoadAllTiffs(self, tiffsPaths : list):
        tiffsImages = list()
        for path in tiffsPaths:
            try:
                image = Image.open(path)
                ncols, nrows = image.size
                nlayers =  image.n_frames
                if self.isTiffCorrect(ncols, nrows, nlayers):
                    tiffsImages.append(image)
                else:
                    print("Uncorrect '" + path + "' image: image params as not the same as in config!")
                    return list()
            except FileNotFoundError:
                print("ReadTiffStackFile: Error. File not found!")
                return list()
        
        return tiffsImages

    # Функция записи всех *.tiff в удобный для CNN файл (! используется в другом модуле)
    def GenerateModel(self, modelName : str, trainTiffs : list, answerTiffs : list):
        try:
            #with open(modelName + ".model", 'wb') as modelFile:
            with open(modelName + ".model", 'wb') as modelFile:
                # write set sizes: Width, height and layers per image, train set power, then all train images, then all answers images
                modelFile.write(self.modelConfig.width.to_bytes(4, 'big'))
                modelFile.write(self.modelConfig.height.to_bytes(4, 'big'))
                modelFile.write(self.modelConfig.layers.to_bytes(4, 'big'))
                modelFile.write(len(trainTiffs).to_bytes(4, 'big'))
                
                # TODO: Maybe need to write and load images 'dtype'!!!!
                #modelFile.write(np.array(train).dtype)

                for train in trainTiffs:
                    for layer in range(train.n_frames):
                        train.seek(layer)
                        np.array(train).tofile(modelFile)
                        #modelFile.write(np.array(train))

                for answer in answerTiffs:
                    for layer in range(answer.n_frames):
                        answer.seek(layer)
                        np.array(answer).tofile(modelFile)
                        #modelFile.write(np.array(answer))

                print("Succesful generate!")
        except IOError:
            print("Bad file path!")
        return

    def TestReadModel(self, modelName : str, trainTiffs : list, answerTiffs : list):
        try:
            with open(modelName + ".model", 'rb') as modelFile:
                # read set sizes: Width, height and layers per image, train set power, then all train images, then all answers images
                width = int.from_bytes(modelFile.read(4), byteorder='big')
                if (width != self.modelConfig.width):
                    return False

                height = int.from_bytes(modelFile.read(4), byteorder='big')
                if (height != self.modelConfig.height):
                    return False

                layers = int.from_bytes(modelFile.read(4), byteorder='big')
                if (layers != self.modelConfig.layers):
                    return False

                power = int.from_bytes(modelFile.read(4), byteorder='big')
                if (power != len(trainTiffs)):
                    return False

                for train in trainTiffs:
                    for layer in range(train.n_frames):
                        train.seek(layer)
                        trainLayer = np.array(train)
                        readedLayer = np.reshape(np.fromfile(modelFile, dtype="uint8", count=width * height), trainLayer.shape)
                        if not np.array_equal(trainLayer, readedLayer):
                            return False

                for answer in answerTiffs:
                    for layer in range(answer.n_frames):
                        answer.seek(layer)
                        answerLayer = np.array(answer)
                        readedLayer = np.reshape(np.fromfile(modelFile, dtype="uint8", count=width * height), answerLayer.shape)
                        if not np.array_equal(answerLayer, readedLayer):
                            return False

                return True
        except IOError:
            print("Bad file path!")
        return False

    # Функция генерации модели
    def CreateModel(self, modelName = "sample_text"):
        # 1 - обход всех путей *.tiff в директории train и добавление в лист
        trainTiffsPaths = self.FindAllPaths(self.modelConfig.trainSetPath)

        # 2 - обход всех путей *.tiff в директории answer и добавление в лист (затем сравнение с результатом)
        answersTiffsPaths = self.FindAllPaths(self.modelConfig.answerSetPath)
        if len(answersTiffsPaths) != len(trainTiffsPaths):
            print("Uncorrect model data: train set and answers are not the same count!")
            return

        # 3 - проверка всех изображений на корректность введенных параметров
        trainTiffs = self.LoadAllTiffs(trainTiffsPaths)
        answersTiffs = self.LoadAllTiffs(answersTiffsPaths)
        if len(trainTiffs) == 0 or len(answersTiffs) == 0:      # Критерий того, что у нас что то не так - пустой лист картинок
            return

        # 4 - поочередная запись в файл информации: (W, H, N, мощность выборки, запись всех train тифов, запись всех ответов)
        self.GenerateModel(modelName, trainTiffs, answersTiffs)

        # 5 (тестовое) - реализация функции чтения модели и проверки прочитанного с записанным для тестирования
        if self.TestReadModel(modelName, trainTiffs, answersTiffs):
            print("All writes good!")
        else:
            print("Something writes bad!")
        return
