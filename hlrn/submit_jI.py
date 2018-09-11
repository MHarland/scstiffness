#!/usr/bin/env python
import sys, subprocess, os, itertools, numpy as np


# prototype_j.py, prototype_stiffness.py
pars_common = {'prototype_files': ['prototype_stiffness.py','prototype_hlrn.sh'],
               'commands': [['chmod +x'], ['chmod +x', 'msub']],
               'target_files': [],
               # target_file scope:
               'cwd': os.getcwd(),
               'accountname': 'hhp00042',
               'n_nodes': 4,
               'walltime': '4:00:00',
               #j: JosephsonExchange2D, JosephsonExchange3D, JosephsonExchange3DAndersen
               #I: SCStiffness2D, SCStiffness3D, SCStiffness3DAndersen
               'lattice': 'SCStiffness3D',
               'niw': 20,
               'nk': 32,
               'tnn': -1,
               'tnnn': .3,
               'tz': -.15}

pars = []
for fname in sys.argv[1:]:
    pars.append(pars_common.copy())
    pars[-1]['fname'] = fname
    pars[-1]['name'] = pars[-1]['fname'][:-9]+'tz'+str(pars[-1]['tz'])+'_'+pars[-1]['lattice']+'_nk'+str(pars[-1]['nk'])+'_niw'+str(pars[-1]['niw'])+'_nambu'

for par in pars:
    par['n_tasks'] = par['n_nodes'] * 24
    par['pyname'] = 'stiff_'+par['name']+'.py'
    par['jobname'] = par['name']
    par['target_files'] = ['stiff_'+par['name']+'.py', 'stiff_'+par['name']+'.sh']

for par in pars:
    for proto_fname, target_fname, commands in zip(par['prototype_files'], par['target_files'], par['commands']):
        proto = open(proto_fname, 'r')
        target = open(target_fname, 'w')
        for line in proto:
            line = str(line)
            while len(line.split('{%', 1)) > 1:
                split1 = line.split('{%', 1)
                split2 = split1[1].split('%}', 1)
                assert split2[0] in par.keys(), "key "+split2[0]+" not found in par.keys()"
                line = split1[0] + str(par[split2[0]]) + split2[1]
            target.write(line)
        del proto, target
        for command in commands:
            if 'sub' in command:
                jobid_str = subprocess.check_output(command+' '+target_fname, shell = True)
                print par['jobname']+' submitted as '+jobid_str[16]
            else:
                subprocess.call(command+' '+target_fname, shell = True)
