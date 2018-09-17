import numpy as np, itertools as itt

from pytriqs.archive import HDFArchive
from pytriqs.gf.local import BlockGf, GfImFreq, GfImTime, inverse
from pytriqs.utility import mpi

from bethe.h5interface import Storage
from bethe.setups.cdmftsquarelattice import NambuMomentumPlaquetteSetup as Setup

from periodization.dmft import LatticeGreensfunction
from periodization.generic import LocalLatticeGreensfunction

from symmetry import symmetrize
from hoppings import Hopping2D, Hopping3D, Hopping3DAndersen

    
class JosephsonExchangeCommon:
    """
    Mu has to be in electron-dispersion due to nambu-basis
    ..._on_the_fly(=True) is slower, but less memory consuming
    """
    def __init__(self, h5name_cdmft, nk, niw, tnn, tnnn, tz = -.15,
                 rjs = [(0,0,0),(1,0,0),(0,0,1),(1,1,0),(1,0,1),(1,1,1),(2,0,0),(0,0,2),(3,0,0),
                        (0,0,3)],
                 verbose = True, loops = [-1], gk_on_the_fly = True, hk_on_the_fly = True):

        self.h5name_cdmft = h5name_cdmft
        self.parameters = {'nk':nk, 'niw': niw, 'tnn': tnn, 'tnnn': tnnn, 'tz': tz}
        self.verbose = verbose
        self.values = dict()
        
        se_cdmft, mu = self.load_cdmft(h5name_cdmft, niw, loops)
        se_cdmft = self.transf_sp_basis(se_cdmft)
        hop = self.get_hopping(mu, tnn, tnnn, tz)
        se, glat = self.calc_correlation_functions(se_cdmft, hop, nk, gk_on_the_fly, hk_on_the_fly)
        self.calc_values(se, glat, rjs)

    def report(self, text):
        if mpi.is_master_node() and self.verbose:
            print text

    def report_detail(self, text):
        if mpi.is_master_node() and self.verbose > 1:
            print text

    def load_cdmft(self, fname, niw, loops):
        self.report('loading CDMFT results from '+fname+'...')
        sto = Storage(fname)
        se = self.load_g('se_imp_iw', sto, niw, loops)
        mu = sto.load('mu')
        return se, mu

    def load_g(self, name, sto, niw, loops, symmetrize_g = True):
        gimpin = sto.load(name, -1)
        gimp = BlockGf(name_block_generator = [(s, GfImFreq(beta = gimpin.beta, n_points = niw, indices = b.indices)) for s, b in gimpin])
        gimptmp = gimp.copy()
        for loop in loops:
            gimpin = sto.load(name, loop)
            for bn, b in gimpin:
                gtau = GfImTime(indices = b.indices, n_points = 10001, beta = b.beta)
                gtau.set_from_inverse_fourier(gimpin[bn])
                gimptmp[bn].set_from_fourier(gtau)
            gimp += gimptmp/len(loops)
        if symmetrize_g:
            gimp << symmetrize(gimp)
        return gimp

    def transf_sp_basis(self, g):
        setup = Setup(g.beta, 0, 0, 0, 0, 1)
        result = setup.reblock_ksum.transform(g)
        return result
    
    def calc_correlation_functions(self, se_cdmft, hopping, nk, gk_on_the_fly, hk_on_the_fly):
        self.report('calculating lattice correlation functions...')
        se_cdmft.copy()
        mu = 0
        blocknames = [k for k in se_cdmft.indices]
        blockindices = [se_cdmft[bn].indices for bn in blocknames]
        weights_r = [1] * hopping.n_r
        glat = LatticeGreensfunction(blocknames, blockindices, hopping.r, hopping.h_r, nk, se_cdmft, mu, weights_r, gk_on_the_fly = gk_on_the_fly, hk_on_the_fly = hk_on_the_fly)
        return se_cdmft, glat

    def calc_values(self, se, glat, rjs):
        self.report('calculating J...')
        ri = tuple([0]*glat.d)
        niw = int(.5*len(se.mesh))
        gtmp = BlockGf(name_block_generator = [[j, GfImFreq(beta = se.beta, n_points = niw,
                                                            indices = [0])] for j in se.indices])
        j_r = np.zeros([len(rjs),3])
        bn = '0'
        for i_rj, rj in enumerate(rjs):
            self.report_detail('...'+str(rj)+'...')
            gtmp.zero()
            gtmp00 = gtmp[bn][0, 0]
            self.report_detail('calculating g_lat_ij...')
            glatij = glat[ri,rj][bn]
            glatji = glatij
            self.report_detail('calculating scalar products...')
            for i, j, k, l in itt.product(*[range(4)]*4): # could be parallelized
                if j!=k and not((j,k) in [(1,5),(5,1),(2,6),(6,2)]): continue
                if l!=i and not((l,i) in [(1,5),(5,1),(2,6),(6,2)]): continue
                gtmp00 += glatij[0+i,4+j]*se[bn][0+j,4+k]*glatji[0+k,4+l]*se[bn][0+l,4+i]
                gtmp00 -= glatij[0+i,0+j]*se[bn][0+j,4+k]*glatji[4+k,4+l]*se[bn][0+l,4+i]
            if np.sum(np.array(ri)-np.array(rj)) == 0:
                for i, j in itt.product(*[range(4)]*2): # could be parallelized
                    if j!=i and not((j,k) in [(1,5),(5,1),(2,6),(6,2)]): continue
                    gtmp00 += glatij[0+i,4+j]*se[bn][0+j,4+i]
                gtmp00 << -1*gtmp00
            self.values[rj] = gtmp.total_density().real

    def save_to_h5(self, h5name = None, groupname = 'josephsonexchange'):
        if h5name is None:
            self._save(self.h5name_cdmft, groupname)
        else:
            self._save(h5name, groupname)

    def _save(self, h5name, groupname):
        if mpi.is_master_node():
            with HDFArchive(h5name) as h5f:
                if not h5f.is_group(groupname):
                    h5f.create_group(groupname)
                    h5f[groupname]['n'] = 0
                n = h5f[groupname]['n']
                h5f[groupname].create_group(str(n))
                cn = self.__class__.__name__
                h5f[groupname][str(n)]['class_name'] = cn
                h5f[groupname][str(n)]['h5name_cdmft'] = self.h5name_cdmft
                h5f[groupname][str(n)].create_group('parameters')
                for pn, p in self.parameters.items():
                    h5f[groupname][str(n)]['parameters'][pn] = p
                h5f[groupname][str(n)].create_group('result_keys')
                h5f[groupname][str(n)].create_group('result_values')
                for v_, (vn, v) in enumerate(self.values.items()):
                    h5f[groupname][str(n)]['result_keys'][str(v_)] = vn
                    h5f[groupname][str(n)]['result_values'][str(v_)] = v
                h5f[groupname]['n'] += 1
            self.report(cn+' written to '+h5name+'/'+groupname+'/'+str(n))


class JosephsonExchange2D(JosephsonExchangeCommon):

    def __init__(self, h5name_cdmft, nk, niw, tnn, tnnn, tz,
                 rjs = [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0)],
                 verbose = True, loops = [-1], gk_on_the_fly = True, hk_on_the_fly = True):

        tz = 0
        self.h5name_cdmft = h5name_cdmft
        self.parameters = {'nk':nk, 'niw': niw, 'tnn': tnn, 'tnnn': tnnn, 'tz': tz}
        self.verbose = verbose
        self.values = dict()

        se_cdmft, mu = self.load_cdmft(h5name_cdmft, niw, loops)
        se_cdmft = self.transf_sp_basis(se_cdmft)
        hop = self.get_hopping(mu, tnn, tnnn)
        se, glat = self.calc_correlation_functions(se_cdmft, hop, nk, gk_on_the_fly, hk_on_the_fly)
        self.calc_values(se, glat, rjs)

    def get_hopping(self, mu, tnn, tnnn, *args):
        return Hopping2D(mu, tnn, tnnn)


class JosephsonExchange3D(JosephsonExchangeCommon):

    def get_hopping(self, mu, tnn, tnnn, tz):
        return Hopping3D(mu, tnn, tnnn, tz)


class JosephsonExchange3DAndersen(JosephsonExchangeCommon):

    def get_hopping(self, mu, tnn, tnnn, tz):
        return Hopping3DAndersen(mu, tnn, tnnn, tz)
