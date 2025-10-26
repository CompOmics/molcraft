import unittest
import logging


class MolCraftTest(unittest.TestCase):

    original_logging_level = logging.NOTSET 

    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger('molcraft')
        cls.original_logging_level = logger.level
        logger.setLevel(logging.CRITICAL) 

    @classmethod
    def tearDownClass(cls):
        logger = logging.getLogger('molcraft')
        logger.setLevel(cls.original_logging_level)