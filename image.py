import numpy as np
import time
import PIL
from PIL import Image


class NotAnImage(Exception):
    pass 
class NotAnNumpyArray(Exception):
    pass 


class ImageProcessor:

    def __init__(self) -> None:
        self.IMG = 'C:/Users/Nikola Kojadinovic/digital-art-gan/images/Abstract_Image_{}.jpg' 

    def ndarrayToPILImage(self,arr):
        '''Converts np.ndarray class to PIL.Image class'''
        return PIL.Image.fromarray(arr) if isinstance(arr, (np.ndarray)) else None ; raise NotAnNumpyArray("Input must be of type: np.ndarray!")

    def PILImageToNdarray(self, img):
        '''Convert PIL.Image class to np.ndarray class'''
        return np.asarray_chkfinite(img) if isinstance(img, (PIL.Image.Image)) else None ; raise NotAnImage("Input must be of type: PIL.Image!")

    def loadImage(self,imgpath, pil = False):
        '''Return np.ndarray representation of an image, specify pil = True for PIL.Image representation'''
        return PIL.Image.open(imgpath) if pil  else np.asarray_chkfinite(imgpath)

    def showImage(self,img):
        '''Displays the image'''
        img.show() if isinstance(img, PIL.Image.Image) else None ; raise NotAnImage("Input must be of type: PIL.Image!")

    def loadImages(self,num, pil = False):
        '''Loads num images from the image directory. Returns a list of ndarray images, specify pil = True for PIL.Image representation'''
        tick, images = 0, []
        starttime = time.time()
        while tick<num:
            images.append(np.asarray_chkfinite(Image.open(self.IMG.format(tick))) if not pil else PIL.Image.open(self.IMG.format(tick)))
            tick+=1
        endtime = time.time()
        print("{} Images loaded, time: {}".format(num, endtime-starttime))

        return np.array(images, dtype = object)

