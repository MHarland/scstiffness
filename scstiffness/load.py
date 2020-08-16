import h5py, numpy as np


class LoadedResult:
    def __init__(self, fname, groupname = 'scstiffness', class_name = 'SCStiffness3D', nk = 32,
                 niw = 50, tnn = -1, tnnn = 0.3, tz = -.15):
        self.params = {'nk': nk, 'niw': niw, 'tnn': tnn, 'tnnn': tnnn, 'tz': tz}
        self.values = {}
        self.result_loaded = False
        self.class_name = None

        params_found = False
        with h5py.File(fname, 'r') as h5f:
            if groupname in h5f.keys():
                for run_, run in h5f[groupname].items():
                    if run_ == 'n': continue
                    params_found = True
                    
                    if run['class_name'].value != class_name:
                        params_found = False
                    else:
                        for parname, par in run['parameters'].items():
                            if self.params[parname] != par.value:
                                params_found = False
                                break
                    if params_found:
                        self.class_name = run['class_name'].value
                        if 'Exchange' in self.class_name:
                            for r_, r_group in run['result_keys'].items():
                                # this is ugly because r was saved as a tuple rather than an array
                                r = []
                                ri_order = []
                                for ri_, ri in r_group.items():
                                    r.append(ri.value)
                                    ri_order.append(int(ri_))
                                ri_ordered = np.sort(ri_order)
                                assert tuple(ri_order) == tuple(ri_ordered), 'wrong ri order'
                                r = tuple(r)
                                self.values.update({r: run['result_values'][r_].value})
                        else:
                            for res_, res in run['result_keys'].items():
                                self.values.update({run['result_keys'][res_].value: run['result_values'][res_].value})
                        self.result_loaded = True
                        break

    def __getitem__(self, key):
        val = None
        if self.result_loaded:
            val = self.values[key]
        return val
