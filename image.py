from PIL import Image
import numpy as np
import time 


class NotAnImage(Exception):
    pass 
class NotAnNumpyArray(Exception):
    pass 


class ImageProcessor:

    def __init__(self) -> None:
        self.IMG = 'C:/Users/Nikola Kojadinovic/nft/images/Abstract_image_{}.jpg' #absolute path to the images directory

    def ndarrayToPILImage(self,arr):
        '''Converts np.ndarray class to PIL.Image class'''
        return Image.fromarray(arr) if isinstance(arr, (np.ndarray)) else None ; raise NotAnNumpyArray("Input must be of type: np.ndarray!")

    def PILImageToNdarray(self, img):
        '''Convert PIL.Image class to np.ndarray class'''
        return np.asarray_chkfinite(img) if isinstance(img, (Image)) else None ; raise NotAnImage("Input must be of type: PIL.Image!")

    def loadImage(self,imgpath, pil = False):
        '''Return np.ndarray representation of an image, specify pil = True for PIL.Image representation'''
        return Image.open(imgpath) if pil  else np.asarray_chkfinite(imgpath)

    def showImage(self,img):
        '''Displays the image'''
        img.show() if isinstance(img, Image) else None ; raise NotAnImage("Input must be of type: PIL.Image!")

    def loadImages(self,num, pil = False):
        '''Loads num images from the image directory. Returns a list of ndarray images, specify pil = True for PIL.Image representation'''
        starttime, tick, images = time.time(), 0, []
        while tick<num:
            images.append(np.asarray_chkfinite(Image.open(self.IMG.format(tick))) if not pil else Image.open(self.IMG.format(tick)))
            tick+=1
        endtime = time.time()
        print("{} Images loaded, time: {}".format(num, endtime-starttime))

        return images

