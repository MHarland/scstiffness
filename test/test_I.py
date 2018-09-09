import unittest, os, numpy as np

from scstiffness.I import *

class TestSCStiffness(unittest.TestCase):

    def test_3d(self):
        fname = 'b52tnnn0.3u8mu2.3_3d_nambu.h5'
        nk = 4
        niw = 20
        tnn = -1
        tnnn = .3
        tz = -.15

        jjos = SCStiffness3D(fname, nk, niw, tnn, tnnn, tz, verbose = False)
        if mpi.is_master_node():
            print
            for r, j in jjos.values.items():
                print str(r)+': '+str(j)
                
    def test_3dandersen(self):
        fname = 'b52tnnn0.3u8mu2.3_3d_nambu.h5'
        nk = 4
        niw = 20
        tnn = -1
        tnnn = .3
        tz = -.15

        jjos = SCStiffness3DAndersen(fname, nk, niw, tnn, tnnn, tz, verbose = False)
        if mpi.is_master_node():
            print
            for r, j in jjos.values.items():
                print str(r)+': '+str(j)

    def test_2d(self):
        fname = 'b52tnnn0.3u8mu2.3_3d_nambu.h5'
        nk = 4
        niw = 20
        tnn = -1
        tnnn = .3
        tz = 0

        jjos = SCStiffness2D(fname, nk, niw, tnn, tnnn, tz, verbose = False)
        if mpi.is_master_node():
            print
            for r, j in jjos.values.items():
                print str(r)+': '+str(j)
