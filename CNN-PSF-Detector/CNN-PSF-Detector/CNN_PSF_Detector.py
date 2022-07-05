from ModelReader import ModelReader
from CNNModel import CNNModel

def run(modelName):
    mr = ModelReader()
    if not mr.ReadModel(modelName):
        print("Model reading failure. Stop working.")
        return
    return

if __name__ == "__main__":
    run("./datasets/testModel.model")
