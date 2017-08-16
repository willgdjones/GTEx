import unittest
from utils.helpers import *


class HelperTestCase(unittest.TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return


    def filter_features(self):

        filter_features_across_all_patchsizes('Lung', 'retrained', 'median', 500)


if __name__ == '__main__':
    unittest.main()
