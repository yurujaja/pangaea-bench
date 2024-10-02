import unittest


class testPackageImports(unittest.TestCase):
    def test_datasets(self):
        from pangaea import datasets

    def test_foundation_models(self):
        from pangaea import encoders

    def test_segmentors(self):
        from pangaea import decoders

    def test_engine(self):
        from pangaea import engine

    def test_run(self):
        from pangaea import run