from numpy import cos, sin, array, eye

def deps_by_dkx(kx, ky, t, tp):
    de = [[-(t+tp+tp*cos(ky))*sin(kx),-complex(0,1)*cos(kx)*(t+tp+tp*cos(ky)),complex(0,1)*tp*sin(kx)*sin(ky),-tp*cos(kx)*sin(ky)],
          [complex(0,1)*cos(kx)*(t+tp+tp*cos(ky)),(t+tp+tp*cos(ky))*sin(kx),tp*cos(kx)*sin(ky),-complex(0,1)*tp*sin(kx)*sin(ky)],
          [-complex(0,1)*tp*sin(kx)*sin(ky),tp*cos(kx)*sin(ky),(-t+tp+tp*cos(ky))*sin(kx),-complex(0,1)*cos(kx)*(t-tp-tp*cos(ky))],
          [-tp*cos(kx)*sin(ky),complex(0,1)*tp*sin(kx)*sin(ky),complex(0,1)*cos(kx)*(t-tp-tp*cos(ky)),(t-tp-tp*cos(ky))*sin(kx)]]
    return array(de)

def deps_by_dky(kx, ky, t, tp):
    de = [[-(t+tp+tp*cos(kx))*sin(ky),complex(0,1)*tp*sin(kx)*sin(ky),-complex(0,1)*(t+tp+tp*cos(kx))*cos(ky),-tp*cos(ky)*sin(kx)],
          [-complex(0,1)*tp*sin(kx)*sin(ky),(-t+tp+tp*cos(kx))*sin(ky),tp*cos(ky)*sin(kx),complex(0,1)*(-t+tp+tp*cos(kx))*cos(ky)],
          [complex(0,1)*(t+tp+tp*cos(kx))*cos(ky),tp*cos(ky)*sin(kx),(t+tp+tp*cos(kx))*sin(ky),-complex(0,1)*tp*sin(kx)*sin(ky)],
          [-tp*cos(ky)*sin(kx),-complex(0,1)*(-t+tp+tp*cos(kx))*cos(ky),complex(0,1)*tp*sin(kx)*sin(ky),(t-tp-tp*cos(kx))*sin(ky)]]
    return array(de)

def deps_by_dkz(kz, tz):
    return -2*tz*sin(kz)*eye(4)
