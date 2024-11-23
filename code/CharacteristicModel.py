import numpy as np

def CharacteristicModel(img, M, gamma):
    XYZ = np.zeros_like(img, dtype=np.float64)
    XYZ = img**gamma
    XYZ = np.transpose(np.tensordot(M, XYZ, axes=([0], [2])), (1, 2, 0))

    return XYZ
