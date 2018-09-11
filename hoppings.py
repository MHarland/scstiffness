import numpy as np
from numpy import cos, sin, array, eye, conj


class Hopping2D:
    """
    h_r and r(translations) are lists with consistent orders
    """
    def __init__(self, mu, tnn, tnnn):
        self.h_r = []
        self.r = [[0,0],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
        self.n_r = len(self.r)
        
        t, s = tnn, tnnn
        h_r = [{'0':[[-mu,t,t,s],[t,-mu,s,t],[t,s,-mu,t],[s,t,t,-mu]]},
               {'0':[[0,t,0,s],[0,0,0,0],[0,s,0,t],[0,0,0,0]]},
               {'0':[[0,0,0,s],[0,0,0,0],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,t,s],[0,0,s,t],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,0,0],[0,0,s,0],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,0,0],[t,0,s,0],[0,0,0,0],[s,0,t,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[0,0,0,0],[s,0,0,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[t,s,0,0],[s,t,0,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[0,s,0,0],[0,0,0,0]]}]
        p3 = np.array([[1,0],[0,-1]]) # particle hole transform dispersion and mu
        u = .5*np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]) # K transform
        for i_h, h in enumerate(h_r):
            self.h_r.append({'0': np.kron(p3, u.dot(np.array(h['0'])).dot(u))})

    def deps_by_dkx(self, k, t, tp, tz):
        kx, ky = k[0], k[1]
        de = [[-(t+tp+tp*cos(ky))*sin(kx),-complex(0,1)*cos(kx)*(t+tp+tp*cos(ky)),
               complex(0,1)*tp*sin(kx)*sin(ky),-tp*cos(kx)*sin(ky)],
              [complex(0,1)*cos(kx)*(t+tp+tp*cos(ky)),(t+tp+tp*cos(ky))*sin(kx),
               tp*cos(kx)*sin(ky),-complex(0,1)*tp*sin(kx)*sin(ky)],
              [-complex(0,1)*tp*sin(kx)*sin(ky),tp*cos(kx)*sin(ky),
               (-t+tp+tp*cos(ky))*sin(kx),-complex(0,1)*cos(kx)*(t-tp-tp*cos(ky))],
              [-tp*cos(kx)*sin(ky),complex(0,1)*tp*sin(kx)*sin(ky),
               complex(0,1)*cos(kx)*(t-tp-tp*cos(ky)),(t-tp-tp*cos(ky))*sin(kx)]]
        return array(de)

    def deps_by_dky(self, k, t, tp, tz):
        kx, ky = k[0], k[1]
        de = [[-(t+tp+tp*cos(kx))*sin(ky),complex(0,1)*tp*sin(kx)*sin(ky),
               -complex(0,1)*(t+tp+tp*cos(kx))*cos(ky),-tp*cos(ky)*sin(kx)],
              [-complex(0,1)*tp*sin(kx)*sin(ky),(-t+tp+tp*cos(kx))*sin(ky),
               tp*cos(ky)*sin(kx),complex(0,1)*(-t+tp+tp*cos(kx))*cos(ky)],
              [complex(0,1)*(t+tp+tp*cos(kx))*cos(ky),tp*cos(ky)*sin(kx),
               (t+tp+tp*cos(kx))*sin(ky),-complex(0,1)*tp*sin(kx)*sin(ky)],
              [-tp*cos(ky)*sin(kx),-complex(0,1)*(-t+tp+tp*cos(kx))*cos(ky),
               complex(0,1)*tp*sin(kx)*sin(ky),(t-tp-tp*cos(kx))*sin(ky)]]
        return array(de)


class Hopping3D(Hopping2D):
    
    def __init__(self, mu, tnn, tnnn, tz):
        self.h_r = []
        self.r = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[-1,1,0],[-1,0,0],[-1,-1,0],[0,-1,0],[1,-1,0],[0,0,1],[0,0,-1]]
        self.n_r = len(self.r)
        
        t, s = tnn, tnnn
        h_r = [{'0':[[-mu,t,t,s],[t,-mu,s,t],[t,s,-mu,t],[s,t,t,-mu]]},
               {'0':[[0,t,0,s],[0,0,0,0],[0,s,0,t],[0,0,0,0]]},
               {'0':[[0,0,0,s],[0,0,0,0],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,t,s],[0,0,s,t],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,0,0],[0,0,s,0],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,0,0],[t,0,s,0],[0,0,0,0],[s,0,t,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[0,0,0,0],[s,0,0,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[t,s,0,0],[s,t,0,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[0,s,0,0],[0,0,0,0]]},
               {'0':[[tz,0,0,0],[0,tz,0,0],[0,0,tz,0],[0,0,0,tz]]},
               {'0':[[tz,0,0,0],[0,tz,0,0],[0,0,tz,0],[0,0,0,tz]]}]
        p3 = np.array([[1,0],[0,-1]]) # particle hole transform dispersion and mu
        u = .5*np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]) # K transform
        for i_h, h in enumerate(h_r):
            self.h_r.append({'0': np.kron(p3, u.dot(np.array(h['0'])).dot(u))})

    def deps_by_dkz(self, k, t, tp, tz):
        kz = k[2]
        return -2*tz*sin(kz)*eye(4)


class Hopping3DAndersen:
    """
    complex interlayer coupling preserving the lattice symmetry
    """
    def __init__(self, mu, tnn, tnnn, tz):
        self.h_r = []
        self.r = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[-1,1,0],[-1,0,0],[-1,-1,0],[0,-1,0],[1,-1,0],
                        [0,0,1],[0,0,-1],
                        [1,1,1],[1,1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],[1,-1,1],[1,-1,-1],
                        [1,0,-1],[1,0,1],[0,1,-1],[0,1,1],[-1,0,-1],[-1,0,1],[0,-1,-1],[0,-1,1]]
        self.n_r = len(self.r)
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
        p3 = np.array([[1,0],[0,-1]]) # particle hole transform dispersion and mu
        u = .5*np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]) # K transform
        for i_h, h in enumerate(h_r):
            self.h_r.append({'0': np.kron(p3, u.dot(np.array(h['0'])).dot(u))})

    def deps_by_dkx(self, k, t, tp, tz):
        kx, ky, kz = k[0], k[1], k[2]
        f00 = -.25 * (4*(t + tp + tp* cos(ky)) + tz * cos(kz)) * sin(kx)
        f01 = -1j* cos(kx)* (t + tp + tp* cos(ky))
        f02 = 1j* tp* sin(kx)* sin(ky)
        f03 = -tp* cos(kx)* sin(ky)
        f11 = .25* (4* (t + tp + tp* cos(ky)) - tz* cos(kz))* sin(kx)
        f12 = tp* cos(kx)* sin(ky)
        f13 = -1j* tp* sin(kx)* sin(ky)
        f22 = -(.25)* (4* t - 4* tp - 4* tp* cos(ky) + tz* cos(kz))* sin(kx)
        f23 = -1j* cos(kx)* (t - tp - tp* cos(ky))
        f33 = -(.25)* (-4* t + 4* tp + 4* tp* cos(ky) + tz* cos(kz))* sin(kx)
        de = [[f00,f01,f02,f03],
              [conj(f01),f11,f12,f13],
              [conj(f02),conj(f12),f22,f23],
              [conj(f03),conj(f13),conj(f23),f33]]
        return array(de)

    def deps_by_dky(self, k, t, tp, tz):
        kx, ky, kz = k[0], k[1], k[2]
        f00 = -(.25)* (4* (t + tp + tp* cos(kx)) + tz* cos(kz))* sin(ky)
        f01 = 1j* tp* sin(kx)* sin(ky)
        f02 = -1j* (t + tp + tp* cos(kx))* cos(ky)
        f03 = -tp* cos(ky)* sin(kx)
        f11 = -(.25)* (4* t - 4* tp - 4* tp* cos(kx) + tz* cos(kz))* sin(ky)
        f12 = tp* cos(ky)* sin(kx)
        f13 = 1j* (-t + tp + tp* cos(kx))* cos(ky)
        f22 = .25* (4* (t + tp + tp* cos(kx)) - tz* cos(kz))* sin(ky)
        f23 = -1j* tp* sin(kx)* sin(ky)
        f33 = -(.25)* (-4* t + 4* tp + 4* tp* cos(kx) + tz* cos(kz))* sin(ky)
        de = [[f00,f01,f02,f03],
              [conj(f01),f11,f12,f13],
              [conj(f02),conj(f12),f22,f23],
              [conj(f03),conj(f13),conj(f23),f33]]
        return array(de)

    def deps_by_dkz(self, k, t, tp, tz):
        kx, ky, kz = k[0], k[1], k[2]
        return -(.25)* tz* (2 + cos(kx) + cos(ky))* sin(kz) * eye(4)
