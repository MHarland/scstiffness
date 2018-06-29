import h5py, sys

for fname in sys.argv[1:]:
    print 'loading '+fname+'...'
    with h5py.File(fname, 'a') as h5f:
        for key in h5f.keys():
            if key in ['scstiffness_rho', 'scstiffness_rho3d', 'scstiffness_rho3dinterlayer']:
                del h5f[key]
                print key, ' removed'
print 'done'
