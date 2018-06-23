from numpy import cos, sin, array, eye, conj

def deps_by_dkx(kx, ky, kz, t, tp, tz):
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

def deps_by_dky(kx, ky, kz, t, tp, tz):
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

def deps_by_dkz(kx, ky, kz, t, tp, tz):
    return -(.25)* tz* (2 + cos(kx) + cos(ky))* sin(kz) * eye(4)
