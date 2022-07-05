import numpy as np

# Class witch provides dataset models reading
class ModelReader:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.layers = 0
        self.setPower = 0
        self.train = list()
        self.answers = list()
        return

    # IMPLEMENTED FROM TESTED "DataSetMaker" MODULE!
    def ReadModel(self, modelName : str):
        try:
            with open(modelName, 'rb') as modelFile:
                # read set sizes: Width, height and layers per image, train set power, then all train images, then all answers images
                self.width = int.from_bytes(modelFile.read(4), byteorder='big')
                self.height = int.from_bytes(modelFile.read(4), byteorder='big')
                self.layers = int.from_bytes(modelFile.read(4), byteorder='big')
                self.setPower = int.from_bytes(modelFile.read(4), byteorder='big')

                for train in range(self.setPower):
                    readedTrain = np.reshape(np.fromfile(modelFile, dtype="uint8", count=self.width * self.height * self.layers), newshape=(self.width, self.height, self.layers))
                    self.train.append(readedTrain)

                for answer in range(self.setPower):
                    readedAnswer = np.reshape(np.fromfile(modelFile, dtype="uint8", count=self.width * self.height * self.layers), newshape=(self.width, self.height, self.layers))
                    self.answers.append(readedAnswer)

                return True
        except IOError:
            print("Bad model name (or path)!")
        return False
