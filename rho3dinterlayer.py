import unittest, os, numpy as np, h5py, itertools as itt
from pytriqs.gf.local import BlockGf, GfImFreq, GfImTime, inverse, iOmega_n
from pytriqs.utility import mpi
from bethe.h5interface import Storage
from bethe.setups.cdmftsquarelattice import NambuMomentumPlaquetteSetup as Setup
from periodization.dmft import LatticeGreensfunction

from mpiLists import scatter_list, allgather_list
from epsderivativeinterlayer import deps_by_dkx, deps_by_dky, deps_by_dkz
from symmetry import symmetrize

    
class SCStiffness:
    def __init__(self, fname, nk, niw, tnn, tnnn, tz = -.15, verbose = True, xx = True, xy = False, xz = False, zz = True, loops = [-1]):
        self.verbose = verbose
        self.report('loading '+fname+'...')
        sto = Storage(fname)
        seimpin = sto.load('se_imp_iw', -1)
        beta = seimpin.beta
        seimp = BlockGf(name_block_generator = [(s, GfImFreq(beta = beta, n_points = niw, indices = b.indices)) for s, b in seimpin])
        seimptmp = seimp.copy()
        gimpin = sto.load('g_imp_iw', -1)
        gimp = BlockGf(name_block_generator = [(s, GfImFreq(beta = beta, n_points = niw, indices = b.indices)) for s, b in gimpin])
        gimptmp = gimp.copy()
        for loop in loops:
            seimpin = sto.load('se_imp_iw', loop)
            for s, b in seimpin:
                gtau = GfImTime(indices = b.indices, n_points = 10001, beta = beta)
                gtau.set_from_inverse_fourier(seimpin[s])
                seimptmp[s].set_from_fourier(gtau)
            seimp += seimptmp/len(loops)
            gimpin = sto.load('g_imp_iw', loop)
            for s, b in gimpin:
                gtau = GfImTime(indices = b.indices, n_points = 10001, beta = beta)
                gtau.set_from_inverse_fourier(gimpin[s])
                gimptmp[s].set_from_fourier(gtau)
            gimp += gimptmp/len(loops)
        
        seimp << symmetrize(seimp)
        gimp << symmetrize(gimp)
        
        mu = sto.load('mu', loop)
        setup = Setup(beta, mu, 999, tnn, tnnn, nk)

        seimp = setup.reblock_ksum.transform(seimp)
        gimp = setup.reblock_ksum.transform(gimp)

        blocknames = [k for k in seimp.indices]
        blockindices = [seimp[bn].indices for bn in blocknames]
        translations = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[-1,1,0],[-1,0,0],[-1,-1,0],[0,-1,0],[1,-1,0],
                        [0,0,1],[0,0,-1],
                        [1,1,1],[1,1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],[1,-1,1],[1,-1,-1],
                        [1,0,-1],[1,0,1],[0,1,-1],[0,1,1],[-1,0,-1],[-1,0,1],[0,-1,-1],[0,-1,1]]
        t, s = tnn, tnnn
        u,v,w = tz, tz/4., -tz/2.
        h_r = [{'0':[[-mu,t,t,s],[t,-mu,s,t],[t,s,-mu,t],[s,t,t,-mu]]},
               {'0':[[0,t,0,s],[0,0,0,0],[0,s,0,t],[0,0,0,0]]},
               {'0':[[0,0,0,s],[0,0,0,0],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,t,s],[0,0,s,t],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,0,0],[0,0,s,0],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,0,0],[t,0,s,0],[0,0,0,0],[s,0,t,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[0,0,0,0],[s,0,0,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[t,s,0,0],[s,t,0,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[0,s,0,0],[0,0,0,0]]},
               
               {'0':[[u,0,0,w],[0,u,w,0],[0,w,u,0],[w,0,0,u]]},
               {'0':[[u,0,0,w],[0,u,w,0],[0,w,u,0],[w,0,0,u]]},
               
               {'0':[[0,0,0,w],[0,0,0,0],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,0,w],[0,0,0,0],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,0,0],[0,0,w,0],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,0,0],[0,0,w,0],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[0,0,0,0],[w,0,0,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[0,0,0,0],[w,0,0,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[0,w,0,0],[0,0,0,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[0,w,0,0],[0,0,0,0]]},

               {'0':[[v,0,0,w],[0,v,0,0],[0,w,v,0],[0,0,0,v]]},
               {'0':[[v,0,0,w],[0,v,0,0],[0,w,v,0],[0,0,0,v]]},
               {'0':[[v,0,0,w],[0,v,w,0],[0,0,v,0],[0,0,0,v]]},
               {'0':[[v,0,0,w],[0,v,w,0],[0,0,v,0],[0,0,0,v]]},
               {'0':[[v,0,0,0],[0,v,w,0],[0,0,v,0],[w,0,0,v]]},
               {'0':[[v,0,0,0],[0,v,w,0],[0,0,v,0],[w,0,0,v]]},
               {'0':[[v,0,0,0],[0,v,0,0],[0,w,v,0],[w,0,0,v]]},
               {'0':[[v,0,0,0],[0,v,0,0],[0,w,v,0],[w,0,0,v]]}
        ]
        mu = 0
        # particle hole transform dispersion and mu
        p3 = np.array([[1,0],[0,-1]])
        # K transform dispersion and mu
        u = .5*np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])

        for i_h, h in enumerate(h_r):
            h_r[i_h] = {'0': np.kron(p3, u.dot(np.array(h['0'])).dot(u))}
        weights_r = [1] * len(translations)
        glat = LatticeGreensfunction(blocknames, blockindices, translations, h_r, nk, seimp, mu, weights_r, gk_on_the_fly = True)

        rhoxx = GfImFreq(beta = beta, n_points = niw, indices = [0])
        rhoxx00 = rhoxx[0,0]
        rhoxy = GfImFreq(beta = beta, n_points = niw, indices = [0])
        rhoxy00 = rhoxy[0,0]
        rhoxz = GfImFreq(beta = beta, n_points = niw, indices = [0])
        rhoxz00 = rhoxy[0,0]
        rhoxz = GfImFreq(beta = beta, n_points = niw, indices = [0])
        rhoxz00 = rhoxz[0,0]
        rhozz = GfImFreq(beta = beta, n_points = niw, indices = [0])
        rhozz00 = rhozz[0,0]
        dgdkx = GfImFreq(beta = beta, n_points = niw, indices = range(8))
        dgdky = GfImFreq(beta = beta, n_points = niw, indices = range(8))
        dgdkz = GfImFreq(beta = beta, n_points = niw, indices = range(8))
        
        glatk = scatter_list(glat.k)
        nk_core = len(glatk)
        glatwk = scatter_list(glat.wk)
        twopi = np.pi*2
        seimp0 = seimp['0']

        for (i_k, k), wk in zip(enumerate(glatk), glatwk):
            self.report('i_k = '+str(i_k+1)+'/'+str(nk_core))
            depsargs = [k[0]*twopi,k[1]*twopi, k[2]*twopi,tnn,tnnn,tz]
            depsdkx = np.kron(p3, deps_by_dkx(*depsargs))
            depsdky = np.kron(p3, deps_by_dky(*depsargs))
            depsdkz = np.kron(p3, deps_by_dkz(*depsargs))
            dgdkx.zero()
            dgdky.zero()
            dgdkz.zero()
            glatgki = glat.gk[i_k]['0']
            for i, j, m, n in itt.product(*[range(8)]*4):
                if i >= 4 and n < 4: continue # anomalous symmetry
                if j >= 4 and m < 4: continue # no anomalous dispersion
                if m >= 4 and j < 4: continue
                dgdkx[i,n] += glatgki[i,j] * depsdkx[j,m] * glatgki[m,n]
                dgdky[i,n] += glatgki[i,j] * depsdky[j,m] * glatgki[m,n]
                dgdkz[i,n] += glatgki[i,j] * depsdkz[j,m] * glatgki[m,n]
            for i, j, k, l in itt.product(*[range(4)]*4):
                if j!=k and not((j,k) in [(1,5),(5,1),(2,6),(6,2)]): continue
                if l!=i and not((l,i) in [(1,5),(5,1),(2,6),(6,2)]): continue
                if xx:
                    rhoxx00 -= .5*dgdkx[0+i,0+j]*seimp0[0+j,4+k]*dgdkx[4+k,4+l]*seimp0[0+l,4+i]
                    rhoxx00 -= .5*dgdkx[4+i,4+j]*seimp0[0+j,4+k]*dgdkx[0+k,0+l]*seimp0[0+l,4+i]
                    rhoxx00 += dgdkx[0+i,4+j]*seimp0[0+j,4+k]*dgdkx[0+k,4+l]*seimp0[0+l,4+i]
                if xy:
                    rhoxy00 -= .5*dgdkx[0+i,0+j]*seimp0[0+j,4+k]*dgdky[4+k,4+l]*seimp0[0+l,4+i]
                    rhoxy00 -= .5*dgdkx[4+i,4+j]*seimp0[0+j,4+k]*dgdky[0+k,0+l]*seimp0[0+l,4+i]
                    rhoxy00 += dgdkx[0+i,4+j]*seimp0[0+j,4+k]*dgdky[0+k,4+l]*seimp0[0+l,4+i]
                if xz:
                    rhoxz00 -= .5*dgdkx[0+i,0+j]*seimp0[0+j,4+k]*dgdkz[4+k,4+l]*seimp0[0+l,4+i]
                    rhoxz00 -= .5*dgdkx[4+i,4+j]*seimp0[0+j,4+k]*dgdkz[0+k,0+l]*seimp0[0+l,4+i]
                    rhoxz00 += dgdkx[0+i,4+j]*seimp0[0+j,4+k]*dgdkz[0+k,4+l]*seimp0[0+l,4+i]
                if zz:
                    rhozz00 -= .5*dgdkz[0+i,0+j]*seimp0[0+j,4+k]*dgdkz[4+k,4+l]*seimp0[0+l,4+i]
                    rhozz00 -= .5*dgdkz[4+i,4+j]*seimp0[0+j,4+k]*dgdkz[0+k,0+l]*seimp0[0+l,4+i]
                    rhozz00 += dgdkz[0+i,4+j]*seimp0[0+j,4+k]*dgdkz[0+k,4+l]*seimp0[0+l,4+i]
        if xx:
            rhoxx << mpi.all_reduce(mpi.world, rhoxx, lambda x, y: x + y)
            rhoxx << rhoxx / (nk**3)
            self.rhoxx = rhoxx.total_density()
        if xy:
            rhoxy << mpi.all_reduce(mpi.world, rhoxy, lambda x, y: x + y)
            rhoxy << rhoxy / (nk**3)
            self.rhoxy = rhoxy.total_density()
        if xz:
            rhoxz << mpi.all_reduce(mpi.world, rhoxz, lambda x, y: x + y)
            rhoxz << rhoxz / (nk**3)
            self.rhoxz = rhoxz.total_density()
        if zz:
            rhozz << mpi.all_reduce(mpi.world, rhozz, lambda x, y: x + y)
            rhozz << rhozz / (nk**3)
            self.rhozz = rhozz.total_density()

    def report(self, text):
        if mpi.is_master_node() and self.verbose:
            print text

    def report_detail(self, text):
        if mpi.is_master_node() and self.verbose > 1:
            print text
