import numpy as np
from PIL import Image
import tifffile as tff          #  https://pypi.org/project/tifffile/


def FindColor(image_tiff):
    colorMode = -1
    for x in image_tiff:
        for y in x:
            if y[0] != 0:
                colorMode = 0       # red
                break
            if y[1] != 0:
                colorMode = 1       # green
                break
            if y[2] != 0:
                colorMode = 2       # blue
                break
    return colorMode

def ReadTiffStackFile(fileName):
    """Function ReadTiffStackFile() reads tiff stack from file and return np.array"""
    print("Loading Image from tiff stack file..... ")
    try:
        "Map to find out how much bits spends on each channel"
        MODE_TO_BPP = {'1':1, 'L':8, 'P':8, 
                    'RGB':8, 'RGBA':8, 'CMYK':8,
                    'YCbCr':8, 'I':32, 'F':32, 
                    "I;16": 16, "I;16B": 16, "I;16L": 16, 
                    "I;16S": 16, "I;16BS": 16, "I;16LS": 16, 
                    "I;32": 32, "I;32B": 32, "I;32L": 32, 
                    "I;32S": 32, "I;32BS": 32, "I;32LS": 32}

        image_tiff = Image.open(fileName)
        ncols, nrows = image_tiff.size
        nlayers =  image_tiff.n_frames
        colorDepth = MODE_TO_BPP[image_tiff.mode]
        imgArray = np.ndarray([nlayers,nrows,ncols])
        
        colorMode = -1
        for i in range(nlayers):
            image_tiff.seek(i)
            
            if (len(np.array(image_tiff).shape) == 2):
                imgArray[i,:,:] = np.array(image_tiff)
            else:
                if (colorMode == -1):
                    colorMode = FindColor(np.array(image_tiff))
                imgArray[i,:,:] = np.array(image_tiff)[:, :, 0 if colorMode == -1 else colorMode]

        print("Done!")
        return imgArray, colorDepth
    except FileNotFoundError:
        print("ReadTiffStackFile: Error. File not found!")
        return 0, 0

def ReadTiffStackFileTFF(fileName):
    """Function ReadTiffStackFile() reads tiff stack from file and return np.array"""
    print("Loading Image from tiff stack file..... ", end = ' ')
    try:
        image_stack = tff.imread(fileName)
        print("Done.")
        return image_stack
    except FileNotFoundError:
        print("ReadTiffStackFileTFF: Error. File not found!")
        return 0



def SaveTiffFiles(tiffDraw = np.zeros([3,4,6]), dirName = "img", filePrefix = ""):
  """ Print files for any input arrray of intensity values
      tiffDraw - numpy ndarray of intensity values"""
  layerNumber = tiffDraw.shape[0]
  for i in range(layerNumber):
    im = Image.fromarray(tiffDraw[i,:,:])
    im.save(dirName+"\\"+filePrefix+str(i).zfill(2)+".tiff")





def SaveTiffStack(tiffDraw = np.zeros([3,4,6]), dirName = "img", filePrefix = "!stack", colorDepth = 8):
    """ Print files for any input arrray of intensity values 
        tiffDraw - numpy ndarray of intensity values"""
    print("trying to save file")
    path = dirName+"\\"+filePrefix+".tif"
    imlist = []
    
    if (colorDepth == 8):
        safeType = 'uint8' 
    elif (colorDepth == 16):  
        safeType = 'uint16'
    else:
        safeType = 'uint32'

    for tmp in tiffDraw:
        imlist.append(Image.fromarray(tmp.astype(safeType)))

    imlist[0].save( path, save_all=True, append_images=imlist[1:])
    print("file saved in one tiff",dirName+"\\"+filePrefix+".tiff")





def SaveTiffStackTFF(tiffDraw = np.zeros([3,4,6]), dirName = "img", filePrefix = "!stack"):
    """ Print files for any input arrray of intensity values
      tiffDraw - numpy ndarray of intensity values"""
    print("trying to save file")
    outTiff = np.rint(tiffDraw).astype('uint16')
    print("outTiff type: ",tiffDraw.dtype)
#    tff.imwrite(dirName+"\\"+filePrefix+".tiff", outTiff)
    tff.imwrite(dirName+"\\"+filePrefix+".tiff", tiffDraw, dtype=tiffDraw.dtype)
    print("file saved in one tiff",dirName+"\\"+filePrefix+".tiff")
