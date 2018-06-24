import numpy as np
from pytriqs.gf.local import BlockGf, GfImFreq
from pytriqs.utility import mpi
from pytriqs.archive import HDFArchive
# rho3dinterlayer, rho3d
from scstiffness.{%lattice%} import SCStiffness


fname = {%fname%}
nk = {%nk%}
niw = {%niw%}
tnn = {%tnn%}
tnnn = {%tnnn%}
tz = {%tz%}
lattice = {%lattice%}

rho = SCStiffness(fname, nk, niw, tnn, tnnn, tz)

groupname = 'scstiffness_'+lattice
if mpi.is_master_node():
    with HDFArchive(fname) as h5f:
        if not h5f.is_group(groupname):
            h5f.create_group(groupname)
            h5f[groupname]['n'] = 0
        n = h5f[groupname]['n']
        h5f[groupname].create_group(str(n))
        h5f[groupname][str(n)] = np.array([nk,niw,tnn,tnnn,tz,rho.rhoxx,rho.rhozz])
        h5f[groupname]['n'] += 1
