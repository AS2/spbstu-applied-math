# Class which provides custom metrics for accurency
class CustomAccurencyMetrics():
    # constructor
    def __init__(self):
        return

    # Method which provides counting accurancy (likehood) of two images
    def CountAccuracy(self, pred_image, true_image, eps = 0.001, shape=(36, 36)):
        accuracy = 0
        total_pxls = shape[0] * shape[1]

        for x in range(shape[0]):
            for y in range(shape[1]):
                accuracy = accuracy + (abs(pred_image[0][0][x][y] - true_image[0][0][x][y]) < eps)
        
        return float(accuracy / total_pxls) * 100

