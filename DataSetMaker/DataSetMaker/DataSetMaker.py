from ModelConfig import ModelConfig
from ModelCreator import ModelCreator

def run(configPath = "\config.txt"):
    modelConfig = ModelConfig(configPath)
    if (modelConfig.isInited):
        modelCreator = ModelCreator(modelConfig)
        modelCreator.CreateModel("testModel")
    return

if __name__ == '__main__':
    run("config.txt")
