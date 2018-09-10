import numpy as np
from scstiffness.j import {%lattice%} as JosephsonExchange


fname = '{%fname%}'
nk = {%nk%}
niw = {%niw%}
tnn = {%tnn%}
tnnn = {%tnnn%}
tz = {%tz%}

jjos = JosephsonExchange(fname, nk, niw, tnn, tnnn, tz)
jjos.save_to_h5()
