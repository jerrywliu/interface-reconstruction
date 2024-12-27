import math
import numpy as np

from geoms import getDistance, getArea, getPolyLineArea, getPolyLineIntersects, lerp
from linear_facet import getLinearFacet, getLinearFacetFromNormal
from circular_facet import getArcFacet, getArcFacetSigmoidRoot

from facet import getNormal, LinearFacet, ArcFacet

def local_reconstruction(opolys, areas, threshold, interfaceReconstruction='Youngs+ours', makeGapless=True, config='normal'):

    predfacets = [[[] for _ in range(len(opolys[0]))] for _ in range(len(opolys))]
    neighbors = [[[None, None] for _ in range(len(opolys[0]))] for _ in range(len(opolys))]
    
    for x in range(len(opolys)):
        for y in range(len(opolys[0])):
            #Interface reconstruction step

            if areas[x][y] < threshold or areas[x][y] > 1-threshold:
                #Full cell, no facet needed
                predfacets[x][y] = []
            else:
                #Mixed cell
                def helperArea(x, y):
                    if x < 0 or y < 0 or x >= len(opolys) or y >= len(opolys[0]):
                        return 0
                    return areas[x][y]

                #Normal calculation
                if interfaceReconstruction == 'Youngs':
                    def getYoungsNormal(x, y):
                        #Youngs normal
                        youngs_alpha = 2
                        f_e = 1/(2+youngs_alpha)*(helperArea(x+1, y-1) + youngs_alpha*helperArea(x+1, y) + helperArea(x+1, y+1))
                        f_w = 1/(2+youngs_alpha)*(helperArea(x-1, y-1) + youngs_alpha*helperArea(x-1, y) + helperArea(x-1, y+1))
                        f_n = 1/(2+youngs_alpha)*(helperArea(x-1, y+1) + youngs_alpha*helperArea(x, y+1) + helperArea(x+1, y+1))
                        f_s = 1/(2+youngs_alpha)*(helperArea(x-1, y-1) + youngs_alpha*helperArea(x, y-1) + helperArea(x+1, y-1))
                        normal = [(f_e - f_w)/2, (f_n - f_s)/2]
                        normal_magnitude = getDistance([0,0], normal)
                        normal = [normal[0]/normal_magnitude, normal[1]/normal_magnitude]

                        return normal

                    l1, l2 = getLinearFacetFromNormal(opolys[x][y], areas[x][y], getYoungsNormal(x, y), threshold)
                    linearfacet = LinearFacet(l1, l2)
                    predfacets[x][y].append(linearfacet)

                elif interfaceReconstruction == 'ELVIRA':
                    def getElviraNormal(x, y):
                        #ELVIRA normal
                        s = [0 for _ in range(6)]
                        n = [None for _ in range(24)]
                        lines = [None for _ in range(24)]
                        l2s = [0 for _ in range(24)]

                        for i in range(-1, 2):
                            s[0] += helperArea(x, y+i) - helperArea(x-1, y+i)
                            s[1] += 0.5*(helperArea(x+1, y+i) - helperArea(x-1, y+i))
                            s[2] += helperArea(x+1, y+i) - helperArea(x, y+i)
                            s[3] += helperArea(x+i, y) - helperArea(x+i, y-1)
                            s[4] += 0.5*(helperArea(x+i, y+1) - helperArea(x+i, y-1))
                            s[5] += helperArea(x+i, y+1) - helperArea(x+i, y)

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

                        """

                        print_areas = [[0 for _ in range(3)] for _ in range(3)]
                        for i in range(3):
                            for j in range(3):
                                print_areas[i][j] = helperArea(x+i-1, y+j-1)
                        print("Print areas: {}".format(print_areas))
                        print("Normals:")
                        print(s)
                        print(n)
                        """
                        
                        curmin = 0
                        if True: #use l2 norm
                            for option in range(24):
                                #print(option)
                                l1, l2 = getLinearFacetFromNormal(opolys[x][y], areas[x][y], n[option], threshold)
                                for i in range(-1, 2):
                                    for j in range(-1, 2):
                                        if not(x < 0 or y < 0 or x >= len(opolys) or y >= len(opolys[0])) and l2s[option] <= l2s[curmin]:
                                            #this is a valid neighbor square
                                            #print(getPolyLineArea(opolys[x+i][y+j], l1, l2)/getArea(opolys[x+i][y+j]))
                                            l2s[option] += (areas[x+i][y+j] - getPolyLineArea(opolys[x+i][y+j], l1, l2)/getArea(opolys[x+i][y+j]))**2

                                if l2s[option] < l2s[curmin]:
                                    curmin = option
                                    
                        else: #use l-inf norm
                            for option in range(12):
                                l1, l2 = getLinearFacetFromNormal(opolys[x][y], areas[x][y], n[option], threshold)
                                for i in range(-1, 2):
                                    for j in range(-1, 2):
                                        if not(x < 0 or y < 0 or x >= len(opolys) or y >= len(opolys[0])):
                                            linf = (areas[x+i][y+j] - getPolyLineArea(opolys[x+i][y+j], l1, l2)/getArea(opolys[x+i][y+j]))
                                            if linf > l2s[curmin]:
                                                break
                                            else:
                                                l2s[option] = max(l2s[option], linf)

                                if l2s[option] < l2s[curmin]:
                                    curmin = option

                        #if x == 3 and y == 2:
                        #    print(1/0)
                        return n[curmin]

                    l1, l2 = getLinearFacetFromNormal(opolys[x][y], areas[x][y], getElviraNormal(x, y), threshold)
                    linearfacet = LinearFacet(l1, l2)
                    predfacets[x][y].append(linearfacet)

                elif interfaceReconstruction == 'Youngs+ours':
                    def getYoungsNormal(x, y):
                        #Youngs normal
                        youngs_alpha = 2
                        f_e = 1/(2+youngs_alpha)*(helperArea(x+1, y-1) + youngs_alpha*helperArea(x+1, y) + helperArea(x+1, y+1))
                        f_w = 1/(2+youngs_alpha)*(helperArea(x-1, y-1) + youngs_alpha*helperArea(x-1, y) + helperArea(x-1, y+1))
                        f_n = 1/(2+youngs_alpha)*(helperArea(x-1, y+1) + youngs_alpha*helperArea(x, y+1) + helperArea(x+1, y+1))
                        f_s = 1/(2+youngs_alpha)*(helperArea(x-1, y-1) + youngs_alpha*helperArea(x, y-1) + helperArea(x+1, y-1))
                        normal = [(f_e - f_w)/2, (f_n - f_s)/2]
                        normal_magnitude = getDistance([0,0], normal)
                        normal = [normal[0]/normal_magnitude, normal[1]/normal_magnitude]

                        return normal

                    #condition on number of mixed neighbors
                    def getMixedNeighbors(x, y):
                        check_neighbors = [[1, 0], [0, -1], [-1, 0], [0, 1]] #clockwise
                        mixed_neighbors = []
                        for i_check_neighbor, check_neighbor in enumerate(check_neighbors):
                            if helperArea(x+check_neighbor[0], y+check_neighbor[1]) > threshold and helperArea(x+check_neighbor[0], y+check_neighbor[1]) < 1-threshold:
                                mixed_neighbors.append(i_check_neighbor)
                        return mixed_neighbors

                    #condition on number of mixed neighbors with exactly 2 mixed neighbors
                    def getMixedCertainNeighbors(x, y):
                        check_neighbors = [[1, 0], [0, -1], [-1, 0], [0, 1]] #clockwise
                        mixed_neighbors = []
                        for i_check_neighbor, check_neighbor in enumerate(check_neighbors):
                            if helperArea(x+check_neighbor[0], y+check_neighbor[1]) > threshold and helperArea(x+check_neighbor[0], y+check_neighbor[1]) < 1-threshold and len(getMixedNeighbors(x+check_neighbor[0], y+check_neighbor[1])) == 2:
                                mixed_neighbors.append(i_check_neighbor)
                        return mixed_neighbors

                    #Get orientation from mixed neighbors: can use mixed neighbors or mixed certain neighbors
                    def getOrientation(x, y, mixed_neighbors):
                        check_neighbors = [[1, 0], [0, -1], [-1, 0], [0, 1]] #clockwise
                        ambiguous_neighbors = True
                        if len(mixed_neighbors) == 2:
                            #2 neighbors, determine orientation based on full neighbors
                            if abs(mixed_neighbors[0] - mixed_neighbors[1]) == 2:
                                #Opposite mixed cells
                                left_test_neighbor = (mixed_neighbors[0]+1) % 4
                                right_test_neighbor = (left_test_neighbor+2) % 4
                                if helperArea(x+check_neighbors[left_test_neighbor][0], y+check_neighbors[left_test_neighbor][1]) > 1-threshold and helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) < threshold:
                                    #Left is full
                                    prev_x = x+check_neighbors[mixed_neighbors[0]][0]
                                    prev_y = y+check_neighbors[mixed_neighbors[0]][1]
                                    next_x = x+check_neighbors[mixed_neighbors[1]][0]
                                    next_y = y+check_neighbors[mixed_neighbors[1]][1]
                                    ambiguous_neighbors = False
                                elif helperArea(x+check_neighbors[left_test_neighbor][0], y+check_neighbors[left_test_neighbor][1]) < threshold and helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) > 1-threshold:
                                    #Right is full
                                    prev_x = x+check_neighbors[mixed_neighbors[1]][0]
                                    prev_y = y+check_neighbors[mixed_neighbors[1]][1]
                                    next_x = x+check_neighbors[mixed_neighbors[0]][0]
                                    next_y = y+check_neighbors[mixed_neighbors[0]][1]
                                    ambiguous_neighbors = False
                            else:
                                #Adjacent mixed cells
                                if (mixed_neighbors[0]+1) % 4 == mixed_neighbors[1]:
                                    mid_test_neighbor = mixed_neighbors[0] #mid_test_neighbor and neighbor clockwise from it are mixed
                                else:
                                    mid_test_neighbor = mixed_neighbors[1]
                                right_test_neighbor = (mid_test_neighbor-1) % 4
                                opp_test_neighbor = (mid_test_neighbor+2) % 4
                                if helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) > 1-threshold and helperArea(x+check_neighbors[opp_test_neighbor][0], y+check_neighbors[opp_test_neighbor][1]) > 1-threshold:
                                    #Both full
                                    prev_x = x+check_neighbors[(mid_test_neighbor+1) % 4][0]
                                    prev_y = y+check_neighbors[(mid_test_neighbor+1) % 4][1]
                                    next_x = x+check_neighbors[mid_test_neighbor][0]
                                    next_y = y+check_neighbors[mid_test_neighbor][1]
                                    ambiguous_neighbors = False
                                elif helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) < threshold and helperArea(x+check_neighbors[opp_test_neighbor][0], y+check_neighbors[opp_test_neighbor][1]) < threshold:
                                    #Both empty
                                    prev_x = x+check_neighbors[mid_test_neighbor][0]
                                    prev_y = y+check_neighbors[mid_test_neighbor][1]
                                    next_x = x+check_neighbors[(mid_test_neighbor+1) % 4][0]
                                    next_y = y+check_neighbors[(mid_test_neighbor+1) % 4][1]
                                    ambiguous_neighbors = False

                        if not(ambiguous_neighbors):
                            return prev_x, prev_y, next_x, next_y
                        else:
                            return None, None, None, None

                    #Get orientation from mixed neighbors: can use mixed neighbors or mixed certain neighbors
                    def getOrientationWeak(x, y, mixed_neighbors):
                        check_neighbors = [[1, 0], [0, -1], [-1, 0], [0, 1]] #clockwise
                        ambiguous_neighbors = True
                        if len(mixed_neighbors) == 2:
                            #2 neighbors, determine orientation based on full neighbors
                            if abs(mixed_neighbors[0] - mixed_neighbors[1]) == 2:
                                #Opposite mixed cells
                                left_test_neighbor = (mixed_neighbors[0]+1) % 4
                                right_test_neighbor = (left_test_neighbor+2) % 4
                                if helperArea(x+check_neighbors[left_test_neighbor][0], y+check_neighbors[left_test_neighbor][1]) > 1-threshold or helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) < threshold:
                                    #Left is full
                                    prev_x = x+check_neighbors[mixed_neighbors[0]][0]
                                    prev_y = y+check_neighbors[mixed_neighbors[0]][1]
                                    next_x = x+check_neighbors[mixed_neighbors[1]][0]
                                    next_y = y+check_neighbors[mixed_neighbors[1]][1]
                                    ambiguous_neighbors = False
                                elif helperArea(x+check_neighbors[left_test_neighbor][0], y+check_neighbors[left_test_neighbor][1]) < threshold or helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) > 1-threshold:
                                    #Right is full
                                    prev_x = x+check_neighbors[mixed_neighbors[1]][0]
                                    prev_y = y+check_neighbors[mixed_neighbors[1]][1]
                                    next_x = x+check_neighbors[mixed_neighbors[0]][0]
                                    next_y = y+check_neighbors[mixed_neighbors[0]][1]
                                    ambiguous_neighbors = False
                            else:
                                #Adjacent mixed cells
                                if (mixed_neighbors[0]+1) % 4 == mixed_neighbors[1]:
                                    mid_test_neighbor = mixed_neighbors[0] #mid_test_neighbor and neighbor clockwise from it are mixed
                                else:
                                    mid_test_neighbor = mixed_neighbors[1]
                                right_test_neighbor = (mid_test_neighbor-1) % 4
                                opp_test_neighbor = (mid_test_neighbor+2) % 4
                                if helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) > 1-threshold or helperArea(x+check_neighbors[opp_test_neighbor][0], y+check_neighbors[opp_test_neighbor][1]) > 1-threshold:
                                    #Both full
                                    prev_x = x+check_neighbors[(mid_test_neighbor+1) % 4][0]
                                    prev_y = y+check_neighbors[(mid_test_neighbor+1) % 4][1]
                                    next_x = x+check_neighbors[mid_test_neighbor][0]
                                    next_y = y+check_neighbors[mid_test_neighbor][1]
                                    ambiguous_neighbors = False
                                elif helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) < threshold or helperArea(x+check_neighbors[opp_test_neighbor][0], y+check_neighbors[opp_test_neighbor][1]) < threshold:
                                    #Both empty
                                    prev_x = x+check_neighbors[mid_test_neighbor][0]
                                    prev_y = y+check_neighbors[mid_test_neighbor][1]
                                    next_x = x+check_neighbors[(mid_test_neighbor+1) % 4][0]
                                    next_y = y+check_neighbors[(mid_test_neighbor+1) % 4][1]
                                    ambiguous_neighbors = False

                        if not(ambiguous_neighbors):
                            return prev_x, prev_y, next_x, next_y
                        else:
                            return None, None, None, None

                    ambiguous_neighbors = True
                    mixed_neighbors = getMixedNeighbors(x, y)
                    prev_x, prev_y, next_x, next_y = getOrientationWeak(x, y, mixed_neighbors)
                    if prev_x is not None:
                        ambiguous_neighbors = False
                    else:
                        #attempt with only unambiguous neighbors
                        mixed_neighbors = getMixedCertainNeighbors(x, y)
                        prev_x, prev_y, next_x, next_y = getOrientationWeak(x, y, mixed_neighbors)
                        if prev_x is not None:
                            ambiguous_neighbors = False
                        else:
                            print(opolys[x][y])

                    if not(ambiguous_neighbors):
                        #Save neighbors
                        neighbors[x][y] = [[prev_x, prev_y], [next_x, next_y]]

                        #Try to fit a linear facet
                        l1, l2 = getLinearFacet(opolys[prev_x][prev_y], opolys[next_x][next_y], areas[prev_x][prev_y], areas[next_x][next_y], threshold)
                        if (config == 'normal' and abs(areas[x][y] - getPolyLineArea(opolys[x][y], l1, l2)/getArea(opolys[x][y])) < threshold):
                            #Area in central poly matches the line, form linear facet
                            #print("Linear facet")
                            linearfacetintersects = getPolyLineIntersects(opolys[x][y], l1, l2)
                            if len(linearfacetintersects) < 2:
                                #Line doesn't intersect target cell (probably extreme area fraction in target cell), default to Youngs normal
                                ambiguous_neighbors = True
                            else:
                                linearfacet = LinearFacet(linearfacetintersects[0], linearfacetintersects[-1])
                                predfacets[x][y].append(linearfacet)
                        elif config == 'linear':
                            normal = [(-l2[1]+l1[1])/getDistance(l1, l2), (l2[0]-l1[0])/getDistance(l1, l2)]
                            try:
                                l1, l2 = getLinearFacetFromNormal(opolys[x][y], areas[x][y], normal, threshold)
                            except:
                                print("getLinearFacetFromNormal({}, {}, {}, {})".format(opolys[x][y], areas[x][y], normal, threshold))
                            linearfacet = LinearFacet(l1, l2)
                            predfacets[x][y].append(linearfacet)
                        else:
                            #Try to fit a circular facet
                            try:
                                #arccenter, arcradius, arcintersects = getArcFacet(opolys[prev_x][prev_y], opolys[x][y], opolys[next_x][next_y], areas[prev_x][prev_y], areas[x][y], areas[next_x][next_y], threshold)
                                arccenter, arcradius, arcintersects = getArcFacetSigmoidRoot(opolys[prev_x][prev_y], opolys[x][y], opolys[next_x][next_y], areas[prev_x][prev_y], areas[x][y], areas[next_x][next_y], threshold)
                                #arccenter, arcradius, arcintersects = getArcFacetNewton2(opolys[prev_x][prev_y], opolys[x][y], opolys[next_x][next_y], areas[prev_x][prev_y], areas[x][y], areas[next_x][next_y], threshold)
                                
                                if arccenter is not None and arcradius is not None and arcintersects is not None:
                                    #Successful circular facet
                                    #print("Circular facet")
                                    arcfacet = ArcFacet(arccenter, arcradius, arcintersects[0], arcintersects[-1])
                                    predfacets[x][y].append(arcfacet)
                                else:
                                    #Failed circular facet, default to Youngs normal
                                    ambiguous_neighbors = True
                            except:
                                #arccenter, arcradius, arcintersects = getArcFacetNewton2(opolys[prev_x][prev_y], opolys[x][y], opolys[next_x][next_y], areas[prev_x][prev_y], areas[x][y], areas[next_x][next_y], threshold)
                                
                                print("getArcFacetNewton2({}, {}, {}, {}, {}, {}, {})".format(opolys[prev_x][prev_y], opolys[x][y], opolys[next_x][next_y], areas[prev_x][prev_y], areas[x][y], areas[next_x][next_y], threshold))
                                ambiguous_neighbors = True

                    #Note that failed circular facet runs go here
                    if ambiguous_neighbors:
                        #Ambiguous neighbors, default to Youngs normal
                        #print("Ambiguity: {}, {}".format(x, y))
                        try:
                            l1, l2 = getLinearFacetFromNormal(opolys[x][y], areas[x][y], getYoungsNormal(x, y), threshold)
                        except:
                            print("getLinearFacetFromNormal({}, {}, {}, {})".format(opolys[x][y], areas[x][y], getYoungsNormal(x, y), threshold))
                        linearfacet = LinearFacet(l1, l2)
                        predfacets[x][y].append(linearfacet)
                        
    #Make C0 continuous
    if makeGapless:
        for x in range(len(opolys)):
            for y in range(len(opolys[0])):
                #Check if left neighbor is well defined
                if areas[x][y] > threshold and areas[x][y] < 1-threshold and neighbors[x][y][0] is not None:
                    leftNeighbor = neighbors[x][y][0]
                    if neighbors[leftNeighbor[0]][leftNeighbor[1]][1] is not None and neighbors[leftNeighbor[0]][leftNeighbor[1]][1][0] == x and neighbors[leftNeighbor[0]][leftNeighbor[1]][1][1] == y:
                        verifyValidLeftNeighbor = True
                    else:
                        verifyValidLeftNeighbor = False

                    if verifyValidLeftNeighbor:
                        #Valid left neighbor, enforce C0 continuity
                        leftNeighbor_pRight = predfacets[leftNeighbor[0]][leftNeighbor[1]][0].pRight
                        cur_pLeft = predfacets[x][y][0].pLeft
                        if getDistance(leftNeighbor_pRight, cur_pLeft) > 0.1:
                            print(getDistance(leftNeighbor_pRight, cur_pLeft))
                        update_endpoint = lerp(leftNeighbor_pRight, cur_pLeft, 0.5)
                        predfacets[x][y][0] = predfacets[x][y][0].update_endpoints(update_endpoint, predfacets[x][y][0].pRight)
                        predfacets[leftNeighbor[0]][leftNeighbor[1]][0] = predfacets[leftNeighbor[0]][leftNeighbor[1]][0].update_endpoints(predfacets[leftNeighbor[0]][leftNeighbor[1]][0].pLeft, update_endpoint)

                    else:
                        #Left neighbor exists but not consistent
                        print("Neighbor endpoints don't match: {}, {}".format([x, y], leftNeighbor))

    return predfacets