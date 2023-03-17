import unittest
#import sys
#sys.path.append("/fh/scratch/delete90/sun_w/si_liu/aTCR/DePTH_package_prepare/src")

class TestImport(unittest.TestCase):
    def test_import_version(self):
        import DePTH
        self.assertTrue(isinstance(DePTH.__version__, str))
        self.assertNotEqual(len(DePTH.__version__), 0)

    def test_import_submodule(self):
        from DePTH import _utils
        self.assertTrue(callable(_utils.get_data))
        self.assertTrue(callable(_utils.SaveBestWeights))


if __name__ == '__main__':
    unittest.main()
