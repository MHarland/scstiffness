import unittest
from pytriqs.utility import mpi #initializes MPI, required on some clusters

from test_j import TestJosephsonExchange
from test_I import TestSCStiffness


suite = unittest.TestSuite()

suite.addTest(TestJosephsonExchange("test_3d"))
suite.addTest(TestJosephsonExchange("test_3dandersen"))
suite.addTest(TestJosephsonExchange("test_2d"))
suite.addTest(TestSCStiffness("test_3d"))
suite.addTest(TestSCStiffness("test_3dandersen"))
suite.addTest(TestSCStiffness("test_2d"))

unittest.TextTestRunner(verbosity = 2).run(suite)
