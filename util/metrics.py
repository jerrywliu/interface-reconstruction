from main.structs.meshes.base_mesh import BaseMesh
from util.initialize_areas import initializeCircle, zalesak, xpluso
from util.initialize_velocity import initializeVelocity

# Get final areas for vortex, deformation, zalesak, xpluso tests
def trueFinalAreas(m: BaseMesh, test_setting="vortex", t=2):
    if test_setting == "vortex":
        fractions = initializeCircle(m, [50.01, 75.01], 15)
    if test_setting == "deformation":
        fractions = initializeCircle(m, [50.01, 75.01], 15)
    elif test_setting == "zalesak":
        fractions = zalesak(m)
    elif test_setting == "xpluso":
        velocity = initializeVelocity(m, t, test_setting=test_setting)
        dx = velocity(0, [0,0])[0]*t # constant velocity for x+o test with same x and y components
        fractions = xpluso(m, dx)

    return fractions

def L2ErrorFractions(final, true):
    assert len(final) == len(true) and len(final[0]) == len(true[0])
    l2_error = 0
    count = 0
    for x in range(len(final)):
        for y in range(len(final[0])):
            if (final[x][y] < 1 and final[x][y] > 0) or (true[x][y] < 1 and true[x][y] > 0):
                l2_error += (final[x][y]-true[x][y])**2
                count += 1
    return l2_error/count, count

def LinfErrorFractions(final, true):
    assert len(final) == len(true) and len(final[0]) == len(true[0])
    linf_error = 0
    count = 0
    for x in range(len(final)):
        for y in range(len(final[0])):
            if (final[x][y] < 1 and final[x][y] > 0) or (true[x][y] < 1 and true[x][y] > 0):
                linf_error = max(linf_error, abs(final[x][y]-true[x][y]))
                count += 1
    return linf_error, count
