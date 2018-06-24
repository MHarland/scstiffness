import numpy as np
from pytriqs.gf.local import BlockGf, GfImFreq
from pytriqs.utility import mpi
from pytriqs.archive import HDFArchive
# j3dinterlayer, j3d
from scstiffness.{%lattice%} import JosephsonExchange


fname = {%fname%}
nk = {%nk%}
niw = {%niw%}
tnn = {%tnn%}
tnnn = {%tnnn%}
tz = {%tz%}
lattice = {%lattice%}

jjos = JosephsonExchange(fname, nk, niw, tnn, tnnn, tz)

groupname = 'josephsonexchange_'+lattice
if mpi.is_master_node():
    with HDFArchive(fname) as h5f:
        if not h5f.is_group(groupname):
            h5f.create_group(groupname)
            h5f[groupname]['n'] = 0
        n = h5f[groupname]['n']
        h5f[groupname].create_group(str(n))
        h5f[groupname][str(n)]['parameters'] = np.array([nk,niw,tnn,tnnn,tz])
        rs = [r for r in jjos.values.keys()]
        js = [jjos.values[r] for r in rs]
        h5f[groupname][str(n)]['translations'] = np.array(rs)
        h5f[groupname][str(n)]['js'] = np.array(js)
        h5f[groupname]['n'] += 1
