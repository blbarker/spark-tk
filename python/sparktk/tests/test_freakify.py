import unittest

from sparktk.tkcontext import freakify


class TestF(unittest.TestCase):

    def test_f(self):
        result = freakify("lunch")
        self.assertEquals("the_freaking_lunch", result)

if __name__ == '__main__':
    unittest.main()
