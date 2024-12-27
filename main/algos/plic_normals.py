import math

from main.geoms.geoms import getArea, getDistance, getPolyLineArea
from main.geoms.linear_facet import getLinearFacetFromNormal

threshold = 1e-10

# Input: stencil = 3x3 array of area fractions
# Output: Young's normal as unit vector
# Assumes Cartesian grid
def getYoungsNormal(stencil):
    def a(poly):
        return 0 if poly is None else poly.getArea()
    youngs_alpha = 2
    f_e = 1/(2+youngs_alpha)*(a(stencil[2][0]) + youngs_alpha*a(stencil[2][1]) + a(stencil[2][2]))
    f_w = 1/(2+youngs_alpha)*(a(stencil[0][0]) + youngs_alpha*a(stencil[0][1]) + a(stencil[0][2]))
    f_n = 1/(2+youngs_alpha)*(a(stencil[0][2]) + youngs_alpha*a(stencil[1][2]) + a(stencil[2][2]))
    f_s = 1/(2+youngs_alpha)*(a(stencil[0][0]) + youngs_alpha*a(stencil[1][0]) + a(stencil[2][0]))
    normal = [(f_e - f_w)/2, (f_n - f_s)/2]
    normal_magnitude = getDistance([0,0], normal)
    if normal_magnitude < 1e-8:
        print([[a(stencil[i][j]) for j in range(len(stencil[0]))] for i in range(len(stencil))])
    normal = [normal[0]/normal_magnitude, normal[1]/normal_magnitude]

    return normal

# Input: stencil = 3x3 array of area fractions
# Output: Young's normal as unit vector
# TODO l-inf
def getLVIRANormal(stencil, norm="l-2"):
    def a(poly): # area fraction
        return 0 if poly is None else poly.getFraction()
    
    s = [0 for _ in range(6)]
    n = [None for _ in range(24)]
    lines = [None for _ in range(24)]
    l2s = [0 for _ in range(24)]

    for i in range(-1, 2):
        s[0] += a(stencil[1][1+i]) - a(stencil[0][1+i])
        s[1] += 0.5*(a(stencil[2][1+i]) - a(stencil[0][1+i]))
        s[2] += a(stencil[2][1+i]) - a(stencil[1][1+i])
        s[3] += a(stencil[1+i][1]) - a(stencil[1+i][0])
        s[4] += 0.5*(a(stencil[1+i][2]) - a(stencil[1+i][0]))
        s[5] += a(stencil[1+i][2]) - a(stencil[1+i][1])

    n[0] = [s[0]/math.sqrt(1+s[0]**2), -1/math.sqrt(1+s[0]**2)]
    n[1] = [s[1]/math.sqrt(1+s[1]**2), -1/math.sqrt(1+s[1]**2)]
    n[2] = [s[2]/math.sqrt(1+s[2]**2), -1/math.sqrt(1+s[2]**2)]
    n[3] = [-1/math.sqrt(1+s[3]**2), s[3]/math.sqrt(1+s[3]**2)]
    n[4] = [-1/math.sqrt(1+s[4]**2), s[4]/math.sqrt(1+s[4]**2)]
    n[5] = [-1/math.sqrt(1+s[5]**2), s[5]/math.sqrt(1+s[5]**2)]

    for i in range(6):
        n[i+6] = [-n[i][0], -n[i][1]]
        n[i+12] = [-n[i][0], n[i][1]]
        n[i+18] = [n[i][0], -n[i][1]]
    
    curmin = 0
    if norm == "l-2": #use l2 norm
        for option in range(24):
            #print(option)
            l1, l2 = getLinearFacetFromNormal(stencil[1][1].points, a(stencil[1][1]), n[option], threshold)
            for i in range(3):
                for j in range(3):
                    if stencil[i][j] is not None and l2s[option] <= l2s[curmin]:
                        #this is a valid neighbor square
                        #print(getPolyLineArea(opolys[x+i][y+j], l1, l2)/getArea(opolys[x+i][y+j]))
                        l2s[option] += (a(stencil[i][j]) - getPolyLineArea(stencil[i][j].points, l1, l2)/stencil[i][j].getMaxArea())**2

            if l2s[option] < l2s[curmin]:
                curmin = option
                
    # elif norm == "l-inf": #use l-inf norm
    #     for option in range(12):
    #         l1, l2 = getLinearFacetFromNormal(opolys[x][y], areas[x][y], n[option], threshold)
    #         for i in range(-1, 2):
    #             for j in range(-1, 2):
    #                 if not(x < 0 or y < 0 or x >= len(opolys) or y >= len(opolys[0])):
    #                     linf = (areas[x+i][y+j] - getPolyLineArea(opolys[x+i][y+j], l1, l2)/getArea(opolys[x+i][y+j]))
    #                     if linf > l2s[curmin]:
    #                         break
    #                     else:
    #                         l2s[option] = max(l2s[option], linf)

    #         if l2s[option] < l2s[curmin]:
    #             curmin = option

    else:
        raise ValueError(f"Unknown norm {norm} in getElviraNormal")

    return n[curmin]