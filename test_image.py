import unittest
import numpy as np 
from image import ImageProcessor, NotAnImage, NotAnNumpyArray
import PIL
from PIL import Image

test_ndarray =np.asarray_chkfinite([[254,124,123],
                            [13,0,135],
                            [156,124,63]])
PIL_image = Image.Image()
img_path = 'C:/Users/Nikola Kojadinovic/digital-art-gan/images/Abstract_image_158.jpg'
test_ndarray_fail = np.asarray_chkfinite([1,1,1])
PIL_image_fail = [2]
img_path_fail = 'asfasf'

class TestImageProcessor(unittest.TestCase):
    
    def test_ndarrayToPILImage(self):
        global test_ndarray
        global test_ndarray_fail
        result = ImageProcessor.ndarrayToPILImage(self, test_ndarray)
        self.assertIsInstance(result, (PIL.Image.Image))
           
    def test_PILImageToNdarray(self):
        global PIL_image
        global PIL_image_fail
        result = ImageProcessor.PILImageToNdarray(self, PIL_image)
        self.assertIsInstance(result, (np.ndarray))
        self.assertRaises(result, NotAnImage, ImageProcessor.PILImageToNdarray, PIL_image_fail )
        
    def test_loadImage(self):
        global img_path
        result = ImageProcessor.loadImage(self,img_path)
        self.assertIsInstance(result, (np.ndarray, PIL.Image.Image))

    def test_showImage(self):
        pass

if __name__ == '__main__':
    unittest.main()
