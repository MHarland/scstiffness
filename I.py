import numpy as np, itertools as itt

from pytriqs.archive import HDFArchive
from pytriqs.gf.local import BlockGf, GfImFreq
from pytriqs.utility import mpi

from j import JosephsonExchangeCommon
from hoppings import Hopping2D, Hopping3D, Hopping3DAndersen
from mpiLists import scatter_list


class SCStiffnessCommon(JosephsonExchangeCommon):
    """
    The theory is gauge invariant, thus J_loc = 0
    """
    def __init__(self, h5name_cdmft, nk, niw, tnn, tnnn, tz = -.15, verbose = True, xx = True, xy = False, xz = False, zz = True, loops = [-1], gk_on_the_fly = True, hk_on_the_fly = True, j_loc = False):

        self.h5name_cdmft = h5name_cdmft
        self.parameters = {'nk':nk, 'niw': niw, 'tnn': tnn, 'tnnn': tnnn, 'tz': tz}
        self.verbose = verbose
        self.values = {}

        se_cdmft, mu = self.load_cdmft(h5name_cdmft, niw, loops)
        se_cdmft = self.transf_sp_basis(se_cdmft)
        self.hop = self.get_hopping(mu, tnn, tnnn, tz)
        se, glat = self.calc_correlation_functions(se_cdmft, self.hop, nk, gk_on_the_fly,
                                                   hk_on_the_fly)
        self.calc_values(se, glat, xx, xy, xz, zz, j_loc)

    def calc_values(self, se, glat, xx, xy, xz, zz, j_loc):
        niw = self.parameters['niw']
        tnn, tnnn, tz = self.parameters['tnn'], self.parameters['tnnn'], self.parameters['tz']
        beta = se.beta

        # init accumulators
        jloc = GfImFreq(beta = beta, n_points = niw, indices = [0])
        jloc00 = jloc[0,0]
        rhoxx = GfImFreq(beta = beta, n_points = niw, indices = [0])
        rhoxx00 = rhoxx[0,0]
        rhoxy = GfImFreq(beta = beta, n_points = niw, indices = [0])
        rhoxy00 = rhoxy[0,0]
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
        glatik = scatter_list(glat.ik)
        twopi = np.pi*2
        p3 = np.array([[1,0],[0,-1]])
        se0 = se['0']

        for i_k, kv, wk in zip(glatik, glatk, glatwk):
            self.report_detail('i_k = '+str(i_k+1)+'/'+str(nk_core))
            depsargs = [kv*twopi,tnn,tnnn,tz]
            if xx or xy or xz:
                depsdkx = np.kron(p3, self.hop.deps_by_dkx(*depsargs))
                dgdkx.zero()
            if xy:
                depsdky = np.kron(p3, self.hop.deps_by_dky(*depsargs))
                dgdky.zero()
            if xz or zz:
                depsdkz = np.kron(p3, self.hop.deps_by_dkz(*depsargs))
                dgdkz.zero()
            
            glatgki = glat.gk[i_k]['0']

            for i, j, m, n in itt.product(*[range(8)]*4):
                if i >= 4 and n < 4: continue # anomalous symmetry
                if j >= 4 and m < 4: continue # no anomalous dispersion
                if m >= 4 and j < 4: continue #
                # only these entries are needed below (also numerically checked)
                if not((i,n) in [(1,5),(1,1),(5,5),(1,6),(2,5),(1,2),(6,5),(2,1),(5,6),(2,6),
                                 (2,2),(6,6)]): continue 
                if xx or xy or xz:
                    dgdkx[i,n] += glatgki[i,j] * depsdkx[j,m] * glatgki[m,n]
                if xy:
                    dgdky[i,n] += glatgki[i,j] * depsdky[j,m] * glatgki[m,n]
                if xz or zz:
                    dgdkz[i,n] += glatgki[i,j] * depsdkz[j,m] * glatgki[m,n]

            # S has only two entries: XX, YY
            for i, j, k, l in [(1,1,1,1),(1,2,2,1),(2,1,1,2),(2,2,2,2)]:
                if xx:
                    rhoxx00 += wk*(dgdkx[0+i,4+j]*se0[0+j,4+k]*dgdkx[0+k,4+l]*se0[0+l,4+i]-
                                   dgdkx[0+i,0+j]*se0[0+j,4+k]*dgdkx[4+k,4+l]*se0[0+l,4+i])
                if xy:
                    rhoxy00 += wk*(dgdkx[0+i,4+j]*se0[0+j,4+k]*dgdky[0+k,4+l]*se0[0+l,4+i]-
                                   dgdkx[0+i,0+j]*se0[0+j,4+k]*dgdky[4+k,4+l]*se0[0+l,4+i]-
                                   dgdky[0+i,0+j]*se0[0+j,4+k]*dgdkx[4+k,4+l]*se0[0+l,4+i])
                if xz:
                    rhoxz00 += wk*(dgdkx[0+i,4+j]*se0[0+j,4+k]*dgdkz[0+k,4+l]*se0[0+l,4+i]-
                                   dgdkx[0+i,0+j]*se0[0+j,4+k]*dgdkz[4+k,4+l]*se0[0+l,4+i]-
                                   dgdkz[0+i,0+j]*se0[0+j,4+k]*dgdkx[4+k,4+l]*se0[0+l,4+i])
                if zz:
                    rhozz00 += wk*(dgdkz[0+i,4+j]*se0[0+j,4+k]*dgdkz[0+k,4+l]*se0[0+l,4+i]-
                                   dgdkz[0+i,0+j]*se0[0+j,4+k]*dgdkz[4+k,4+l]*se0[0+l,4+i])
                if j_loc:
                    jloc00 += wk*(glatgki[0+i,0+j]*se0[0+j,4+k]*glatgki[4+k,4+l]*se0[0+l,4+i]-
                                  glatgki[0+i,4+j]*se0[0+j,4+k]*glatgki[0+k,4+l]*se0[0+l,4+i])
            for i, j in [(1,1),(1,2),(2,1),(2,2)]:
                if j_loc:
                    jloc00 += (-1)*wk*(glatgki[0+i,4+j]*se0[0+j,4+i])
        if xx:
            rhoxx << mpi.all_reduce(mpi.world, rhoxx, lambda x, y: x + y)
            self.values['xx'] = rhoxx.total_density().real
        if xy:
            rhoxy << mpi.all_reduce(mpi.world, rhoxy, lambda x, y: x + y)
            self.values['xy'] = rhoxy.total_density()
        if xz:
            rhoxz << mpi.all_reduce(mpi.world, rhoxz, lambda x, y: x + y)
            self.values['xz'] = rhoxz.total_density()
        if zz:
            rhozz << mpi.all_reduce(mpi.world, rhozz, lambda x, y: x + y)
            self.values['zz'] = rhozz.total_density().real
        if j_loc:
            jloc << mpi.all_reduce(mpi.world, jloc, lambda x, y: x + y)
            self.values['j_loc'] = jloc.total_density().real

    def save_to_h5(self, h5name = None, groupname = 'scstiffness'):
        if h5name is None:
            self._save(self.h5name_cdmft, groupname)
        else:
            self._save(h5name, groupname)


class SCStiffness2D(SCStiffnessCommon):

    def __init__(self, h5name_cdmft, nk, niw, tnn, tnnn, tz = 0, verbose = True, xx = True, xy = False, loops = [-1], gk_on_the_fly = True, hk_on_the_fly = True, j_loc = False):

        tz = 0
        self.h5name_cdmft = h5name_cdmft
        self.parameters = {'nk':nk, 'niw': niw, 'tnn': tnn, 'tnnn': tnnn, 'tz': tz}
        self.verbose = verbose
        self.values = {}

        se_cdmft, mu = self.load_cdmft(h5name_cdmft, niw, loops)
        se_cdmft = self.transf_sp_basis(se_cdmft)
        self.hop = self.get_hopping(mu, tnn, tnnn)
        se, glat = self.calc_correlation_functions(se_cdmft, self.hop, nk, gk_on_the_fly,
                                                   hk_on_the_fly)
        self.calc_values(se, glat, xx, xy, False, False, j_loc)

    def get_hopping(self, mu, tnn, tnnn, *args):
        return Hopping2D(mu, tnn, tnnn)


class SCStiffness3D(SCStiffnessCommon):

    def get_hopping(self, mu, tnn, tnnn, tz):
        return Hopping3D(mu, tnn, tnnn, tz)


class SCStiffness3DAndersen(SCStiffnessCommon):

    def get_hopping(self, mu, tnn, tnnn, tz):
        return Hopping3DAndersen(mu, tnn, tnnn, tz)
