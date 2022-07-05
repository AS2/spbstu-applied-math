class ModelConfig:
    def __init__(self, configPath = "\config.txt"):
        # TODO: parse these params from file
        self.trainSetPath = ""
        self.answerSetPath = ""
        self.width = 36
        self.height = 36
        self.layers = 20
        self.isInited = False

        # parse file
        try:
            with open(configPath, mode='r') as configHandler:
                for line in configHandler:
                    #print(line)
                    sublines = line.split(sep=":")
                    if len(sublines) == 0:
                        continue
                    elif len(sublines) != 2:
                        print("Bad syntax: '" + line + "'")
                        return
                    else:
                        if sublines[0] == "TRAIN-PATH":
                            self.trainSetPath = sublines[1].strip()
                        elif sublines[0] == "ANSWER-PATH":
                            self.answerSetPath = sublines[1].strip()
                        elif sublines[0] == "WIDTH":
                            self.width = int(sublines[1].strip())
                        elif sublines[0] == "HEIGHT":
                            self.height = int(sublines[1].strip())
                        elif sublines[0] == "LAYERS":
                            self.layers = int(sublines[1].strip())
                        else:
                            print("Unknown argument: '" + sublines[0] + "'")
                            return
                self.isInited = True
        except IOError:
            print("Bad config path")
        return
