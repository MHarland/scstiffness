from scstiffness.I import {%lattice%} as SCStiffness


fname = {%fname%}
nk = {%nk%}
niw = {%niw%}
tnn = {%tnn%}
tnnn = {%tnnn%}
tz = {%tz%}

stif = SCStiffness(fname, nk, niw, tnn, tnnn, tz)
stif.save_to_h5()
