import unittest
import numpy as np
from image_helpers import scale_image

class TestImageHelpers(unittest.TestCase):
    def test_scale_image(self):
        image = np.ones((100, 100))
        scaled_image = scale_image(image, scale=0.7)
        self.assertTrue(np.allclose(scaled_image, 0.7))

if __name__ == '__main__':
    unittest.main()