from geoms import getDistance, lerp, getArea, mergePolys, getPolyIntersectArea, getPolyLineArea, getPolyLineIntersects, lineIntersect, getCentroid
from linear_facet import getLinearFacet, getLinearFacetFromNormal
from circular_facet import getCircleIntersectArea, getCircleCircleIntersects, getArcFacet, getArcFacetNewton, getCircleLineIntersects2, getCenter, getCircumcircle, getArcFacetRoot, getArcFacetNewton2
from corner_facet import getPolyCornerArea, getPolyCurvedCornerArea, getCurvedCornerFacet
import math
import random

#opolys = grid of quads
#areas = grid of area fractions per quad
#TODO: add epsilon consideration for merging
def merge(opolys, areas):
    #Merging
    dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    mergedpolys = [] #coordinates of polys to be merged
    predmergedpolys = [[[] for _ in range(len(opolys[0]))] for _ in range(len(opolys))]
    neighborsmergedpolys = [[[] for _ in range(len(opolys[0]))] for _ in range(len(opolys))]

    #Compute initial neighbors
    for x in range(len(opolys)):
        for y in range(len(opolys[0])):
            if areas[x][y] == 0 or areas[x][y] == 1:
                predmergedpolys[x][y] = None
                neighborsmergedpolys[x][y] = None
                continue
            #Compute number of neighbors
            gridsquare = [x, y]
            predmergedpolys[x][y] = [[x, y]]
            for direction in dirs:
                #Check directions in counterclockwise order
                testsquare = [gridsquare[0] + direction[0], gridsquare[1] + direction[1]]
                if testsquare[0] < len(opolys) and testsquare[0] >= 0 and testsquare[1] < len(opolys[0]) and testsquare[1] >= 0 and areas[testsquare[0]][testsquare[1]] < 1 and areas[testsquare[0]][testsquare[1]] > 0:
                    #This direction is a partial area neighbor
                    neighborsmergedpolys[x][y].append(testsquare)
            if len(neighborsmergedpolys[x][y]) == 1:
                print("One: {}".format([x, y]))
    
    #By here, predmergedpolys[x][y] = None if full, [[x, y]] otherwise. neighborsmergedpolys[x][y] = None if full, [neighbors] otherwise.
    #Merge squares with 1 or 3+ neighbors
    for x in range(len(opolys)):
        for y in range(len(opolys[0])):
            gridneighbors = neighborsmergedpolys[x][y]
            if gridneighbors is None or len(gridneighbors) == 2:
                #Full cell or unambiguous cell, skip
                continue
            #Casework on number of neighbors
            assert len(gridneighbors) > 0, "Mixed cell with no mixed neighbors: increase fineness of resolution!"
            if len(gridneighbors) == 1 and not(x == 0 or y == 0 or x == len(opolys)-1 or y == len(opolys[0])-1):
                #Check if diagonal is present
                diagonals = []
                for diagonaldirection in [[1, 1], [-1, 1], [-1, -1], [1, -1]]:
                    diagonalsquare = [x+diagonaldirection[0], y+diagonaldirection[1]]
                    if areas[diagonalsquare[0]][diagonalsquare[1]] > 0 and areas[diagonalsquare[0]][diagonalsquare[1]] < 1 and len(neighborsmergedpolys[diagonalsquare[0]][diagonalsquare[1]]) == 1:
                        diagonals.append(diagonalsquare)
                if len(diagonals) == 1:
                    gridneighbors.append(diagonals[0])
                    neighborsmergedpolys[diagonals[0][0]][diagonals[0][1]].append([x, y])
                    print("Ones: {}, {}".format(diagonals[0], [x, y]))

                else:
                    #No diagonals, maybe long thin strip, probably corner here
                    mergesquares = [[x, y]]
                    curx = gridneighbors[0][0]
                    cury = gridneighbors[0][1]
                    prevx = x
                    prevy = y
                    continueSpike = True
                    while1 = 0
                    while continueSpike and while1 < 50:
                        while1 += 1
                        mergesquares.append([curx, cury])
                        if len(neighborsmergedpolys[curx][cury]) >= 3:
                            continueSpike = False
                            #Update merged cells
                            mergeneighbors = []
                            for mergeneighbor in neighborsmergedpolys[curx][cury]:
                                if mergeneighbor != [prevx, prevy]:
                                    mergeneighbors.append(mergeneighbor)
                            mergecells = []
                            for mergesquare in mergesquares:
                                for mergecell in predmergedpolys[mergesquare[0]][mergesquare[1]]:
                                    if mergecell not in mergecells:
                                        mergecells.append(mergecell)
                            print("1 neighbor case:")
                            print(mergecells)
                            print(mergeneighbors)
                            for mergecell in mergecells:
                                predmergedpolys[mergecell[0]][mergecell[1]] = mergecells
                                neighborsmergedpolys[mergecell[0]][mergecell[1]] = mergeneighbors
                        else:
                            assert len(neighborsmergedpolys[curx][cury]) == 2, "Mixed cells form one cell wide strip: increase fineness of resolution! {}".format([curx, cury])
                            if neighborsmergedpolys[curx][cury][0] == [prevx, prevy]:
                                prevx = curx
                                prevy = cury
                                cursquare = neighborsmergedpolys[curx][cury][1]
                                curx = cursquare[0]
                                cury = cursquare[1]
                            else:
                                prevx = curx
                                prevy = cury
                                cursquare = neighborsmergedpolys[curx][cury][0]
                                curx = cursquare[0]
                                cury = cursquare[1]
                    if while1 == 50:
                        print("While1 failed")
                        print(1/0)

    for x in range(len(opolys)):
        for y in range(len(opolys[0])):
            gridneighbors = neighborsmergedpolys[x][y]
            if gridneighbors is None or len(gridneighbors) <= 2:
                #Full cell or unambiguous cell, skip
                continue
            else:
                #3+ neighbors
                #invariant: mergeneighbors contains cells to be merged, whose neighbors may not have been explored yet; gridneighbors are the cells that need to be explored
                mergecells = []
                for mergecell in predmergedpolys[x][y]:
                    mergecells.append(mergecell)
                mergeneighbors = [[x, y]]
                #5/26/22:
                """
                for gridneighbor in gridneighbors:
                    for gridcell in predmergedpolys[gridneighbor[0]][gridneighbor[1]]:
                        mergecells.append(gridcell)
                    mergeneighbors.append(gridneighbor)
                """
                while len(mergeneighbors) > 0:
                    print("Mergeneighbors")
                    print(mergeneighbors)
                    testsquare = mergeneighbors[0]
                    mergeneighbors.pop(0)
                    testneighbors = neighborsmergedpolys[testsquare[0]][testsquare[1]]
                    if len(testneighbors) >= 3:
                        for testneighbor in testneighbors:
                            if testneighbor not in mergecells:
                                for mergecell in predmergedpolys[testneighbor[0]][testneighbor[1]]:
                                    mergecells.append(mergecell)
                                mergeneighbors.append(testneighbor)
                    elif len(testneighbors) == 2:
                        for testneighbor in testneighbors:
                            if testneighbor not in mergecells:
                                dircount = 0
                                for test2neighbor in neighborsmergedpolys[testneighbor[0]][testneighbor[1]]:
                                    if test2neighbor in mergecells:
                                        dircount += 1
                                if dircount >= 2:
                                    for mergecell in predmergedpolys[testneighbor[0]][testneighbor[1]]:
                                        mergecells.append(mergecell)
                                    mergeneighbors.append(testneighbor)
                #mergecells consists of cells to be merged + the two neighbors (not to be merged)
                #TODO: current code greedily merges all ambiguous cells until ambiguities gone. This can create large, narrow clusters of merged cells, which ideally could be broken into smaller clusters.
                mergeneighbors = []
                for mergecell in mergecells:
                    """
                    nummergeneighbors = 0
                    numallneighbors = len(neighborsmergedpolys[mergecell[0]][mergecell[1]])
                    for testneighbor in neighborsmergedpolys[mergecell[0]][mergecell[1]]:
                        if testneighbor in mergecells:
                            nummergeneighbors += 1
                    if nummergeneighbors < 2 and numallneighbors > 1 and mergecell not in mergeneighbors:
                        #Endpoint
                        mergeneighbors.append(mergecell)
                    """
                    #5/24/22:
                    trackmergeneighbors = []
                    #Compute number of neighbors
                    for direction in dirs:
                        #Check directions in counterclockwise order
                        testsquare = [mergecell[0] + direction[0], mergecell[1] + direction[1]]
                        if testsquare[0] < len(opolys) and testsquare[0] >= 0 and testsquare[1] < len(opolys[0]) and testsquare[1] >= 0 and areas[testsquare[0]][testsquare[1]] < 1 and areas[testsquare[0]][testsquare[1]] > 0:
                            #This direction is a partial area neighbor
                            trackmergeneighbors.append(testsquare)
                    nummergeneighbors = 0
                    for trackmergeneighbor in trackmergeneighbors:
                        if trackmergeneighbor not in mergecells:
                            nummergeneighbors += 1
                    if nummergeneighbors == 1:
                        #Endpoint
                        mergeneighbors.append(mergecell)
                    elif nummergeneighbors == 0:
                        #Normal
                        continue
                    else:
                        #Weird case, throw error
                        print("Two or more remaining neighbors not in the merged cell:")
                        print(mergecells)
                        print(mergeneighbors)
                for mergeneighbor in mergeneighbors:
                    mergecells.remove(mergeneighbor)
                print(mergecells)
                print(mergeneighbors)
                assert len(mergeneighbors) == 2
                #Update merged cells
                for mergecell in mergecells:
                    predmergedpolys[mergecell[0]][mergecell[1]] = mergecells
                    neighborsmergedpolys[mergecell[0]][mergecell[1]] = mergeneighbors
    
    mergedpolyindices = [[None for _ in range(len(opolys[0]))] for _ in range(len(opolys))]
    mergedpolyinfos = []
    mergedcoords = [] #the coordinates of the fully merged poly
    mergedareafractions = [] #area fractions of fully merged poly

    """
    print("Merged polys:")
    def printmodule(x):
        if x is None or len(x) > 1:
            return x
        else:
            return None
    print(list(map(lambda x : list(map(lambda y : printmodule(y), x)), predmergedpolys)))
    """

    for x in range(len(opolys)):
        for y in range(len(opolys[0])):
            if predmergedpolys[x][y] is None or len(predmergedpolys[x][y]) == 0:
                continue
            elif len(predmergedpolys[x][y]) == 1:
                #x, y
                mergedpolyindices[x][y] = len(mergedpolyinfos)
                mergedpolyinfos.append([predmergedpolys[x][y], neighborsmergedpolys[x][y]])
                mergedcoords.append(opolys[x][y])
                mergedareafractions.append(areas[x][y])
            else:
                #cluster
                print("Mergecells")
                print(mergecells)
                mergedpolyinfo = [predmergedpolys[x][y], neighborsmergedpolys[x][y]]
                if mergedpolyinfo not in mergedpolyinfos:
                    mergecells = predmergedpolys[x][y].copy()
                    for mergecell in mergecells:
                        mergedpolyindices[mergecell[0]][mergecell[1]] = len(mergedpolyinfos)
                    mergedpolyinfos.append(mergedpolyinfo)
                    boundary = opolys[mergecells[0][0]][mergecells[0][1]]
                    mergecells.pop(0)
                    while2 = 0
                    while mergecells and while2 < 50:
                        while2 += 1
                        try:
                            print("Merging: boundary, new to be merged")
                            print("{}, {}".format(boundary, opolys[mergecells[0][0]][mergecells[0][1]]))
                            boundary = mergePolys(boundary, opolys[mergecells[0][0]][mergecells[0][1]])
                            mergecells.pop(0)
                        except:
                            mergetroublecell = mergecells[0]
                            mergecells.pop(0)
                            mergecells.append(mergetroublecell)
                    if while2 == 50:
                        print("While2 failing")
                        print(1/0)
                    mergecells = predmergedpolys[x][y].copy()
                    mergedcoords.append(boundary)
                    boundaryarea = sum(list(map(lambda x : areas[x[0]][x[1]]*getArea(opolys[x[0]][x[1]]), mergecells)))/sum(list(map(lambda x : getArea(opolys[x[0]][x[1]]), mergecells)))
                    mergedareafractions.append(boundaryarea)

    return mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions

#mergedpolyindices: xy-grid of merged polygon indices: if mergedpolyindices[x][y] = i, mergedpolyinfos[i] is the merged polygon at [x, y]
#mergedpolyinfos: list of merged polygons: [[list of [x, y]s to be merged], [[neighbor1], [neighbor2]]]
#mergedcoords: list of coords for merged polygons in same order as mergedpolyinfos
#mergedareafractions: list of area fractions for merged polygons in same order as mergedpolyinfos

#mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions: from merging function
#opolys, areas: original volume fractions, used to refine linear facets
#threshold: tolerance for errors in volume fractions

#Hyperparameters:
linearerrorthreshold = 1e-6 #if area fraction error in linear facet < linearerrorthreshold, use linear facet at this cell
cornererrorthreshold = 1e-6 #if area fraction error in corner facet < cornererrorthreshold, use (straight edged) corner facet at this cell
curvaturethreshold = 1e-3 #-float('inf') #30 #if adjacent curvatures > curvaturethreshold, try to fit a curved corner facet
curvedcornerthreshold = 1e-3 #-float('inf') #if area fraction error in curved corner < curvedcornerthreshold, use curved corner at this cell
curvedconvthreshold = 1e-10 #if curved corner optimization is only improving area fractions by < curvedconvthreshold, declare it converged
curvednewtonfactor = 0.1 #how much to step in Newton direction when optimizing curved corners

def makeFacets(mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions, opolys, areas, threshold, makeGapless, useCorners=True, useCircles=True):
    #facets
    predfacets = [[None] * len(opolys[0]) for _ in range(len(opolys))]
    facetsunique = []

    #orientations
    predorientations = [[None] * len(opolys[0]) for _ in range(len(opolys))]
    mergedorientations = [None] * len(mergedpolyinfos)
    
    #compute facet orientations
    for y in range(len(opolys[0])):
        for x in range(len(opolys)):
            if mergedpolyindices[x][y] is not None and mergedorientations[mergedpolyindices[x][y]] == None:
                #finding path, starting with bottom left most point
                #orientation of path: outer boundary = True, inner boundary = False
                if (y > 0 and areas[x][y-1] == 1) or (x > 0 and areas[x-1][y] == 1):
                    curOrientation = False
                else:
                    curOrientation = True
                
                path = []
                curmergedpolyindex = mergedpolyindices[x][y]
                curmergedpoly = mergedpolyinfos[curmergedpolyindex]
                if curmergedpoly[1][0][0] < curmergedpoly[1][1][0]:
                    nextmergedpolyindex = mergedpolyindices[curmergedpoly[1][1][0]][curmergedpoly[1][1][1]]
                elif curmergedpoly[1][0][0] > curmergedpoly[1][1][0]:
                    nextmergedpolyindex = mergedpolyindices[curmergedpoly[1][0][0]][curmergedpoly[1][0][1]]
                else:
                    if curmergedpoly[1][0][1] < curmergedpoly[1][1][1]:
                        nextmergedpolyindex = mergedpolyindices[curmergedpoly[1][0][0]][curmergedpoly[1][0][1]]
                    else:
                        nextmergedpolyindex = mergedpolyindices[curmergedpoly[1][1][0]][curmergedpoly[1][1][1]]
                path.append(curmergedpolyindex)
                mergedorientations[curmergedpolyindex] = curOrientation
                for point in curmergedpoly[0]:
                    predorientations[point[0]][point[1]] = curOrientation
                prevmergedpolyindex = curmergedpolyindex
                while3 = 0
                while nextmergedpolyindex != path[0] and while3 < 200:
                    while3 += 1
                    #Here: handle single neighbor cases
                    if nextmergedpolyindex is not None:
                        curmergedpolyindex = nextmergedpolyindex
                    else:
                        raise Exception("Interface path is unclear: increase fineness of resolution!")
                        print(mergedpolyinfos[curmergedpolyindex])
                    curmergedpoly = mergedpolyinfos[curmergedpolyindex]
                    if curmergedpoly[1][0] in mergedpolyinfos[prevmergedpolyindex][0]:
                        nextmergedpolyindex = mergedpolyindices[curmergedpoly[1][1][0]][curmergedpoly[1][1][1]]
                    else:
                        print(curmergedpoly)
                        print(mergedpolyinfos[prevmergedpolyindex])
                        assert curmergedpoly[1][1] in mergedpolyinfos[prevmergedpolyindex][0], mergedpolyinfos[prevmergedpolyindex]
                        nextmergedpolyindex = mergedpolyindices[curmergedpoly[1][0][0]][curmergedpoly[1][0][1]]
                    path.append(curmergedpolyindex)
                    mergedorientations[curmergedpolyindex] = curOrientation
                    for point in curmergedpoly[0]:
                        predorientations[point[0]][point[1]] = curOrientation
                    prevmergedpolyindex = curmergedpolyindex
                if while3 == 200:
                    print("While3 failed")
                    print(1/0)
                    
                assert len(path) >= 2
                path.append(path[0])
                path.append(path[1])
                
                if not(curOrientation):
                    path.reverse()
                    
                def plotpath(x):
                    if len(mergedcoords[x]) > 4:
                        mergedpolyxys = mergedpolyinfos[x][0]
                        mergedpolyareas = list(map(lambda x : areas[x[0]][x[1]], mergedpolyxys))
                        mergedpolycoords = list(map(lambda x : opolys[x[0]][x[1]], mergedpolyxys))
                        mergedneighbors = mergedpolyinfos[x][-1]
                        mergedneighborareas = list(map(lambda x : areas[x[0]][x[1]], mergedneighbors))
                        mergedneighborcoords = list(map(lambda x : opolys[x[0]][x[1]], mergedneighbors))
                        return [mergedcoords[x], mergedpolyinfos[x], mergedpolyareas, mergedpolycoords, mergedneighborareas, mergedneighborcoords]
                    else:
                        return None

                def plotpath2(x):
                    if len(mergedcoords[x]) > 4:
                        return x
                    else:
                        return None
                        
                #print(list(map(lambda x: mergedcoords[x], path)))
                #print(list(map(plotpath, path)))
                #plotpath2ed = list(map(plotpath2, path))

                facetfitted = [None] * (len(path)-2)
                notUseCircles_linear = [None] * (len(path)-2)
                
                #Make linear facets
                for pathelement in range(1, len(path)-1):
                    #previ, curi, nexti are indices of mergedpolyinfos
                    curi = path[pathelement]
                    previ = path[pathelement-1]
                    nexti = path[pathelement+1]
                    prevpolygon = mergedcoords[previ]
                    prevpolygonarea = mergedareafractions[previ]
                    nextpolygon = mergedcoords[nexti]
                    nextpolygonarea = mergedareafractions[nexti]
                    try:
                        facetline1, facetline2 = getLinearFacet(prevpolygon, nextpolygon, prevpolygonarea, nextpolygonarea, threshold)
                        if abs(mergedareafractions[curi] - getPolyLineArea(mergedcoords[curi], facetline1, facetline2)/getArea(mergedcoords[curi])) > linearerrorthreshold:
                            #maybe excess merging reduced resolution: try immediate neighbors
                            if mergedpolyinfos[curi][1][0] in mergedpolyinfos[previ][0]:
                                prevneighbor = mergedpolyinfos[curi][1][0]
                                nextneighbor = mergedpolyinfos[curi][1][1]
                            else:
                                assert mergedpolyinfos[curi][1][1] in mergedpolyinfos[previ][0]
                                prevneighbor = mergedpolyinfos[curi][1][1]
                                nextneighbor = mergedpolyinfos[curi][1][0]
                            prevpolygon = opolys[prevneighbor[0]][prevneighbor[1]]
                            nextpolygon = opolys[nextneighbor[0]][nextneighbor[1]]
                            prevpolygonarea = areas[prevneighbor[0]][prevneighbor[1]]
                            nextpolygonarea = areas[nextneighbor[0]][nextneighbor[1]]
                            facetline1, facetline2 = getLinearFacet(prevpolygon, nextpolygon, prevpolygonarea, nextpolygonarea, threshold)
                        if abs(mergedareafractions[curi] - getPolyLineArea(mergedcoords[curi], facetline1, facetline2)/getArea(mergedcoords[curi])) < linearerrorthreshold:
                            intersects = getPolyLineIntersects(mergedcoords[curi], facetline1, facetline2)
                            if len(intersects) > 0:
                                facetfitted[pathelement-1] = ['linear', intersects]
                        elif not(useCircles):
                            l1 = facetline1
                            l2 = facetline2
                            normal = [(-l2[1]+l1[1])/getDistance(l1, l2), (l2[0]-l1[0])/getDistance(l1, l2)]
                            facetline1, facetline2 = getLinearFacetFromNormal(mergedcoords[curi], mergedareafractions[curi], normal, threshold)
                            intersects = getPolyLineIntersects(mergedcoords[curi], facetline1, facetline2)
                            if len(intersects) > 0:
                                notUseCircles_linear[pathelement-1] = ['linear', intersects]
                        else:
                            continue
                    except:
                        print("Failed linear facet")

                print("Linear")
                print(facetfitted)
                print(notUseCircles_linear)
                            
                if useCorners:

                    #Make corners
                    #Find closest linear facets to left/right of corner poly
                    lefts = [None] * (len(facetfitted))
                    rights = [None] * (len(facetfitted))
                    curlinearfacet = None
                    for pathelement in range(1, 2*len(facetfitted)-1):
                        if facetfitted[(pathelement) % len(facetfitted)] is None:
                            lefts[(pathelement) % len(facetfitted)] = curlinearfacet
                        else:
                            curlinearfacet = facetfitted[(pathelement) % len(facetfitted)][1]
                    for pathelement in range(1, 2*len(facetfitted)-1):
                        if facetfitted[(len(facetfitted)-1-pathelement) % len(facetfitted)] is None:
                            rights[(len(facetfitted)-1-pathelement) % len(facetfitted)] = curlinearfacet
                        else:
                            curlinearfacet = facetfitted[(len(facetfitted)-1-pathelement) % len(facetfitted)][1]

                    #Try to make corners
                    #here pathelement is 1 less than pathelement from linear facet fitting
                    for pathelement in range(len(facetfitted)):
                        if facetfitted[pathelement] is None and lefts[pathelement] is not None and rights[pathelement] is not None and lefts[pathelement][0] != rights[pathelement][0] and lefts[pathelement][1] != rights[pathelement][1]:
                            prevFacet = lefts[pathelement]
                            nextFacet = rights[pathelement]
                            corner, _, _ = lineIntersect(prevFacet[0], prevFacet[1], nextFacet[0], nextFacet[1])
                            if corner is None:
                                #No feasible intersection, continue #TODO: do a test to see if corner point lies within the two segments
                                continue
                            corner = [prevFacet[0], corner, nextFacet[1]]
                            cornerareafraction = getPolyCornerArea(mergedcoords[path[pathelement]], corner[0], corner[1], corner[2])/getArea(mergedcoords[path[pathelement]])
                            if abs(cornerareafraction - mergedareafractions[path[pathelement]]) < cornererrorthreshold:
                                facetfitted[pathelement] = ['corner', corner]
                            else:
                                continue
                            print(pathelement)
                            print(corner)

                    print("Corners")
                    print(facetfitted)

                #Try to make arc facets for the remaining ones
                if useCircles:

                    for pathelement in range(1, len(path)-1):
                        if facetfitted[pathelement-1] is None:
                            
                            #previ, curi, nexti are indices of mergedpolyinfos
                            curi = path[pathelement]
                            previ = path[pathelement-1]
                            nexti = path[pathelement+1]
                            #maybe excess merging reduced resolution: try immediate neighbors
                            if mergedpolyinfos[curi][1][0] in mergedpolyinfos[previ][0]:
                                prevneighbor = mergedpolyinfos[curi][1][0]
                                nextneighbor = mergedpolyinfos[curi][1][1]
                            else:
                                assert mergedpolyinfos[curi][1][1] in mergedpolyinfos[previ][0]
                                prevneighbor = mergedpolyinfos[curi][1][1]
                                nextneighbor = mergedpolyinfos[curi][1][0]
                            prevpolygon = opolys[prevneighbor[0]][prevneighbor[1]]
                            nextpolygon = opolys[nextneighbor[0]][nextneighbor[1]]
                            prevpolygonarea = areas[prevneighbor[0]][prevneighbor[1]]
                            nextpolygonarea = areas[nextneighbor[0]][nextneighbor[1]]
                            curpolygon = mergedcoords[curi]
                            curpolygonarea = mergedareafractions[curi]
                            #Edit 1/5/21:
                            """
                            prevpolygon = mergedcoords[previ]
                            prevpolygonarea = mergedareafractions[previ]
                            curpolygon = mergedcoords[curi]
                            curpolygonarea = mergedareafractions[curi]
                            nextpolygon = mergedcoords[nexti]
                            nextpolygonarea = mergedareafractions[nexti]
                            """
                            try:
                                arccenter, arcradius, arcintersects = getArcFacet(prevpolygon, curpolygon, nextpolygon, prevpolygonarea, curpolygonarea, nextpolygonarea, threshold)
                                #If failed: skip
                                if arccenter is None or arcradius is None or arcintersects is None:
                                    arccenter, arcradius, arcintersects = getArcFacetNewton2(prevpolygon, curpolygon, nextpolygon, prevpolygonarea, curpolygonarea, nextpolygonarea, threshold)
                                    if arccenter is None or arcradius is None or arcintersects is None:
                                        continue
                                if len(arcintersects) == 0:
                                    print("No intersects: getArcFacet({}, {}, {}, {}, {}, {}, {})".format(prevpolygon, curpolygon, nextpolygon, prevpolygonarea, curpolygonarea, nextpolygonarea, threshold))
                                else:
                                    #print("Successful run: getArcFacet({}, {}, {}, {}, {}, {}, {})".format(prevpolygon, curpolygon, nextpolygon, prevpolygonarea, curpolygonarea, nextpolygonarea, threshold))
                                    facetfitted[pathelement-1] = ['arc', arccenter, arcradius, arcintersects]
                                    #print("Answer: {}".format(facetfitted[pathelement-1]))
                                    """
                                    if facetfitted[(pathelement-2) % len(facetfitted)] is not None and facetfitted[(pathelement-2) % len(facetfitted)][0] == 'arc':
                                        arccenter2, arcradius2, arcintersects2 = getArcFacetNewton(prevpolygon, curpolygon, nextpolygon, prevpolygonarea, curpolygonarea, nextpolygonarea, threshold, facetfitted[(pathelement-2) % len(facetfitted)][1][0], facetfitted[(pathelement-2) % len(facetfitted)][1][1], facetfitted[(pathelement-2) % len(facetfitted)][2])
                                        print("Arc face normal: {}, {}, {}".format(arccenter, arcradius, arcintersects))
                                        print("getArcFacet({}, {}, {}, {}, {}, {}, {})".format(prevpolygon, curpolygon, nextpolygon, prevpolygonarea, curpolygonarea, nextpolygonarea, threshold))
                                        print("Arc face newton: {}, {}, {}".format(arccenter2, arcradius2, arcintersects2))
                                    """
                            except:
                                print("Failed run: getArcFacet({}, {}, {}, {}, {}, {}, {})".format(prevpolygon, curpolygon, nextpolygon, prevpolygonarea, curpolygonarea, nextpolygonarea, threshold))
                                continue

                    print('Arc one')
                    #print(facetfitted)

                    #Retry failed circle facets, Newton's method, with circle facet adjacent guess
                    #i = pathelement - 1

                    for i in range(len(facetfitted)):
                        if facetfitted[i] is None or (facetfitted[i][0] == 'arc' and len(facetfitted[i][3]) == 0):
                            #previ, curi, nexti are indices of mergedpolyinfos
                            curi = path[i+1]
                            previ = path[i]
                            nexti = path[i+2]
                            #maybe excess merging reduced resolution: try immediate neighbors
                            if mergedpolyinfos[curi][1][0] in mergedpolyinfos[previ][0]:
                                prevneighbor = mergedpolyinfos[curi][1][0]
                                nextneighbor = mergedpolyinfos[curi][1][1]
                            else:
                                assert mergedpolyinfos[curi][1][1] in mergedpolyinfos[previ][0]
                                prevneighbor = mergedpolyinfos[curi][1][1]
                                nextneighbor = mergedpolyinfos[curi][1][0]
                            prevpolygon = opolys[prevneighbor[0]][prevneighbor[1]]
                            nextpolygon = opolys[nextneighbor[0]][nextneighbor[1]]
                            prevpolygonarea = areas[prevneighbor[0]][prevneighbor[1]]
                            nextpolygonarea = areas[nextneighbor[0]][nextneighbor[1]]
                            curpolygon = mergedcoords[curi]
                            curpolygonarea = mergedareafractions[curi]
                            #Edit 1/5/21:
                            """
                            prevpolygon = mergedcoords[previ]
                            prevpolygonarea = mergedareafractions[previ]
                            curpolygon = mergedcoords[curi]
                            curpolygonarea = mergedareafractions[curi]
                            nextpolygon = mergedcoords[nexti]
                            nextpolygonarea = mergedareafractions[nexti]
                            """
                            #Try previous to find circle facet for Newton's method
                            if facetfitted[(i-1) % len(facetfitted)] is not None and facetfitted[(i-1) % len(facetfitted)][0] == 'arc' and facetfitted[(i-1) % len(facetfitted)][1] is not None:
                                gcenterx = facetfitted[(i-1) % len(facetfitted)][1][0]
                                gcentery = facetfitted[(i-1) % len(facetfitted)][1][1]
                                gradius = facetfitted[(i-1) % len(facetfitted)][2]
                            #Try next to see if circle facet for Newton's method
                            elif facetfitted[(i+1) % len(facetfitted)] is not None and facetfitted[(i+1) % len(facetfitted)][0] == 'arc' and facetfitted[(i+1) % len(facetfitted)][1] is not None:
                                gcenterx = facetfitted[(i+1) % len(facetfitted)][1][0]
                                gcentery = facetfitted[(i+1) % len(facetfitted)][1][1]
                                gradius = facetfitted[(i+1) % len(facetfitted)][2]
                            gcenterx = None
                            gcentery = None
                            gradius = None
                            print("Running getArcFacetRoot: getArcFacetRoot({}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(prevpolygon, curpolygon, nextpolygon, prevpolygonarea, curpolygonarea, nextpolygonarea, threshold, gcenterx, gcentery, gradius))
                            print(i)
                            arccenter, arcradius, arcintersects = getArcFacetRoot(prevpolygon, curpolygon, nextpolygon, prevpolygonarea, curpolygonarea, nextpolygonarea, threshold, gcenterx, gcentery, gradius)
                            if arccenter is not None and arcradius is not None and arcintersects is not None:
                                facetfitted[i] = ['arc', arccenter, arcradius, arcintersects]
                            continue

                            gcenterx = None
                            gcentery = None
                            gradius = None
                            #Try previous to find circle facet for Newton's method
                            if facetfitted[(i-1) % len(facetfitted)] is not None and facetfitted[(i-1) % len(facetfitted)][0] == 'arc' and facetfitted[(i-1) % len(facetfitted)][1] is not None:
                                gcenterx = facetfitted[(i-1) % len(facetfitted)][1][0]
                                gcentery = facetfitted[(i-1) % len(facetfitted)][1][1]
                                gradius = facetfitted[(i-1) % len(facetfitted)][2]
                            #Try next to see if circle facet for Newton's method
                            elif facetfitted[(i+1) % len(facetfitted)] is not None and facetfitted[(i+1) % len(facetfitted)][0] == 'arc' and facetfitted[(i+1) % len(facetfitted)][1] is not None:
                                gcenterx = facetfitted[(i+1) % len(facetfitted)][1][0]
                                gcentery = facetfitted[(i+1) % len(facetfitted)][1][1]
                                gradius = facetfitted[(i+1) % len(facetfitted)][2]
                            try:
                                print("Newton's method arc facet: {}".format(curi))
                                arccenter, arcradius, arcintersects = getArcFacetNewton(prevpolygon, curpolygon, nextpolygon, prevpolygonarea, curpolygonarea, nextpolygonarea, threshold, gcenterx, gcentery, gradius)
                                facetfitted[i] = ['arc', arccenter, arcradius, arcintersects]
                                print("Newton's method result: {}".format(facetfitted[i]))
                            except:
                                #Circle facet algorithm with Newton's method failed or no nearby circular facets
                                print("getArcFacetNewton({}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(prevpolygon, curpolygon, nextpolygon, prevpolygonarea, curpolygonarea, nextpolygonarea, threshold, gcenterx, gcentery, gradius))
                                pass

                    print("Arc two")
                    #print(facetfitted)
                                
                    #Try to fit a curved corner
                    #printer = list(map(lambda x : x[2], facetfitted))
                    #print(printer)
                    #print(list(map(lambda x : abs(1/printer[(x+1) % len(printer)] - 1/printer[x]), range(len(printer)))))

                    if useCorners:

                        for i in range(len(facetfitted)):
                            curi = path[i+1]
                            previ = path[i]
                            nexti = path[i+2]
                            curpolygon = mergedcoords[curi]
                            if facetfitted[i] is not None and facetfitted[i][0] == 'arc' and not(facetfitted[(i-1) % len(facetfitted)] is not None and (facetfitted[(i-1) % len(facetfitted)][0] == 'curvedcorner' or (facetfitted[(i-1) % len(facetfitted)][0] == 'arc' and abs((1/facetfitted[(i-1) % len(facetfitted)][2]) - (1/facetfitted[i][2])) < curvaturethreshold)) and facetfitted[(i+1) % len(facetfitted)] is not None and (facetfitted[(i+1) % len(facetfitted)][0] == 'curvedcorner' or (facetfitted[(i+1) % len(facetfitted)][0] == 'arc' and abs((1/facetfitted[(i+1) % len(facetfitted)][2]) - (1/facetfitted[i][2])) < curvaturethreshold))):
                                #Local curvatures differ by more than curvaturethreshold: try a curved corner
                                #Find leftward arc neighbor
                                prevarc = (i-1) % len(facetfitted)
                                #Go until you find two arc neighboring arc facets that have similar curvature
                                while prevarc != i and (facetfitted[prevarc] is None or facetfitted[(prevarc-1) % len(facetfitted)] is None or (facetfitted[prevarc][0] != 'arc' and facetfitted[prevarc][0] != 'linear') or (facetfitted[prevarc][0] == 'arc' and facetfitted[(prevarc-1) % len(facetfitted)][0] == 'arc' and abs(facetfitted[prevarc][2]-facetfitted[(prevarc-1) % len(facetfitted)][2]) > curvaturethreshold)): #should be >
                                    prevarc = (prevarc-1) % len(facetfitted)
                                if prevarc == i:
                                    #No neighbors
                                    continue
                                nextarc = prevarc+1
                                #Go until you find two arc neighboring arc facets that have similar curvature
                                while nextarc != prevarc+len(facetfitted) and (facetfitted[nextarc % len(facetfitted)] is None or facetfitted[(nextarc+1) % len(facetfitted)] is None or (facetfitted[nextarc % len(facetfitted)][0] != 'arc' and facetfitted[nextarc % len(facetfitted)][0] != 'linear') or (facetfitted[nextarc % len(facetfitted)][0] == 'arc' and facetfitted[(nextarc+1) % len(facetfitted)][0] == 'arc' and abs(facetfitted[nextarc % len(facetfitted)][2]-facetfitted[(nextarc+1) % len(facetfitted)][2]) > curvaturethreshold)): #should be >
                                    nextarc = (nextarc+1) % len(facetfitted)
                                if nextarc % len(facetfitted) == prevarc:
                                    #Only one neighbor
                                    continue

                                #Widen cluster of curved corner polys by 1 on both sides
                                prevarc = (prevarc-1) % len(facetfitted)
                                nextarc = (nextarc+1) % len(facetfitted)
                                #print('prevarc: {}, nextarc: {}'.format(prevarc, nextarc))
                                
                                prevcornerfacetside = facetfitted[prevarc]
                                nextcornerfacetside = facetfitted[nextarc]
                                #print("{}".format(prevcornerfacetside))
                                #print("{}".format(nextcornerfacetside))

                                if prevcornerfacetside[0] == 'linear' and nextcornerfacetside[0] == 'arc':
                                    prevcenter = None
                                    prevradius = None
                                    curvedcorner1 = prevcornerfacetside[1][0]
                                    nextcenter = nextcornerfacetside[1]
                                    nextradius = nextcornerfacetside[2]
                                    curvedcorner2 = nextcornerfacetside[3][0]

                                    curvedcornerpoint = getCircleLineIntersects2(prevcornerfacetside[1][0], prevcornerfacetside[1][1], nextcenter, nextradius)
                                    if len(curvedcornerpoint) == 1:
                                        curvedcornerpoint = curvedcornerpoint[0]
                                    elif len(curvedcornerpoint) > 1:
                                        if getDistance(curpolygon[0], curvedcornerpoint[0]) >= getDistance(curpolygon[0], curvedcornerpoint[1]):
                                            curvedcornerpoint = curvedcornerpoint[1]
                                        else:
                                            curvedcornerpoint = curvedcornerpoint[0]
                                elif prevcornerfacetside[0] == 'arc' and nextcornerfacetside[0] == 'linear':
                                    prevcenter = prevcornerfacetside[1]
                                    prevradius = prevcornerfacetside[2]
                                    curvedcorner1 = prevcornerfacetside[3][0]
                                    nextcenter = None
                                    nextradius = None
                                    curvedcorner2 = nextcornerfacetside[1][1]

                                    curvedcornerpoint = getCircleLineIntersects2(nextcornerfacetside[1][0], nextcornerfacetside[1][1], prevcenter, prevradius)
                                    if len(curvedcornerpoint) == 1:
                                        curvedcornerpoint = curvedcornerpoint[0]
                                    elif len(curvedcornerpoint) > 1:
                                        if getDistance(curpolygon[0], curvedcornerpoint[0]) >= getDistance(curpolygon[0], curvedcornerpoint[1]):
                                            curvedcornerpoint = curvedcornerpoint[1]
                                        else:
                                            curvedcornerpoint = curvedcornerpoint[0]
                                elif prevcornerfacetside[0] == 'arc' and nextcornerfacetside[0] == 'arc':
                                    try:
                                        prevcenter = prevcornerfacetside[1]
                                        prevradius = prevcornerfacetside[2]
                                        curvedcorner1 = prevcornerfacetside[3][-1]
                                        nextcenter = nextcornerfacetside[1]
                                        nextradius = nextcornerfacetside[2]
                                        curvedcorner2 = nextcornerfacetside[3][0]
                                        curvedcornerpoint = getCircleCircleIntersects(prevcenter, nextcenter, prevradius, nextradius)
                                        if len(curvedcornerpoint) == 0:
                                            continue
                                        if getDistance(curpolygon[0], curvedcornerpoint[0]) >= getDistance(curpolygon[0], curvedcornerpoint[1]):
                                            curvedcornerpoint = curvedcornerpoint[1]
                                        else:
                                            curvedcornerpoint = curvedcornerpoint[0]
                                    except:
                                        #print("Two arc curved corner failed")
                                        continue
                                elif prevcornerfacetside[0] == 'linear' and nextcornerfacetside[0] == 'linear':
                                    #two linears
                                    prevcenter = None
                                    nextcenter = None
                                    prevradius = None
                                    curvedcorner1 = prevcornerfacetside[1][0]
                                    nextradius = None
                                    curvedcorner2 = nextcornerfacetside[1][1]
                                    curvedcornerpoint, _, _ = lineIntersect(prevcornerfacetside[1][0], prevcornerfacetside[1][1], nextcornerfacetside[1][0], nextcornerfacetside[1][1])
                                else:
                                    continue
                                
                                #Use all polys in between two nearest successful facets
                                curvedcornergridindices = list(map(lambda x : path[((x+prevarc) % len(facetfitted))+1], range(((nextarc+1-prevarc) % len(facetfitted)))))
                                #print("Curved corner indices: {}".format(curvedcornergridindices))
                                #print("Curved corner point: {}".format(curvedcornerpoint))
                                if curvedcornerpoint is None or len(curvedcornerpoint) < 2:
                                    #print("Failed corner")
                                    continue
                                curvedcornerpolys = list(map(lambda x : mergedcoords[x], curvedcornergridindices))
                                curvedcornerareas = list(map(lambda x : mergedareafractions[x], curvedcornergridindices[1:-1]))
                                if len(curvedcornerareas) < 1:
                                    #print('Nothing in corner')
                                    continue
                                curvedcornerguessareas = list(map(lambda x : getPolyCurvedCornerArea(mergedcoords[x], curvedcorner1, curvedcornerpoint, curvedcorner2, prevradius, nextradius)/getArea(mergedcoords[x]), curvedcornergridindices[1:-1]))
                                #print(curvedcornerguessareas)
                                #print(curvedcornerpolys)
                                maxcurvedcornererror = list(map(lambda x : abs(curvedcornerareas[x]-curvedcornerguessareas[x]), range(len(curvedcornerareas))))
                                #print(maxcurvedcornererror)
                                #print(prevradius)
                                #print(nextradius)
                                maxcurvedcornererror = max(maxcurvedcornererror)
                                if maxcurvedcornererror < curvedcornerthreshold:
                                    #Initial guess is close enough: optimize with Newton's method
                                    try:
                                        #curvedcorner = [curvedcorner1, curvedcornerpoint, curvedcorner2]
                                        curvedcorner = getCurvedCornerFacet(curvedcornerpolys, curvedcornerareas, curvedcorner1, curvedcornerpoint, curvedcorner2, prevradius, nextradius, threshold, curvedconvthreshold, curvednewtonfactor)
                                    except:
                                        print("Failed: getCurvedCornerFacet({}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(curvedcornerpolys, curvedcornerareas, curvedcorner1, curvedcornerpoint, curvedcorner2, prevradius, nextradius, threshold, curvedconvthreshold, curvednewtonfactor))
                                        curvedcorner = [curvedcorner1, curvedcornerpoint, curvedcorner2]
                                    for corneri in range(((nextarc-prevarc-1) % len(facetfitted))):
                                        cornerindex = (prevarc+1+corneri) % len(facetfitted)
                                        #continue #TODO: remove this to add back curved corners
                                        if mergedorientations[path[i+1]]:
                                            facetfitted[cornerindex] = ['curvedcorner', prevcenter, nextcenter, prevradius, nextradius, curvedcorner.copy()]
                                        else:
                                            facetfitted[cornerindex] = ['curvedcorner', nextcenter, prevcenter, nextradius, prevradius, curvedcorner.copy()]

                        print("Curved corner")
                        def f3(x):
                            if x is None:
                                return "NONE!!!!!"
                            elif x[0] == 'arc':
                                return None
                            else:
                                return x

                        print(list(map(lambda x : f3(x), facetfitted)))

                #Fix unmade facets (fill with arc facets)
                for i in range(len(facetfitted)):
                    if useCircles and (facetfitted[i] is None or (facetfitted[i][0] == 'arc' and len(facetfitted[i][3]) == 0)):
                        try:
                            #This is an unfitted facet, assume the two neighbors are fitted and are arc facets
                            prevfacet = facetfitted[(i-1) % len(facetfitted)]
                            nextfacet = facetfitted[(i+1) % len(facetfitted)]
                            fixedcenter = lerp(prevfacet[1], nextfacet[1], 0.5)
                            fixedradius = (prevfacet[2]+nextfacet[2])/2
                            _, fixedintersects = getCircleIntersectArea(fixedcenter, fixedradius, mergedcoords[path[i]])
                            if getDistance(fixedintersects[0], prevfacet[3][-1]) > getDistance(fixedintersects[1], prevfacet[3][-1]):
                                fixedintersects.reverse()
                            facetfitted[i] = ['arc', fixedcenter, fixedradius, fixedintersects]
                            if facetfitted[i] is None:
                                print("Unpredicted facet, filled with {}".format(facetfitted[i]))
                            else:
                                print("Facet with no intersects, filled with {}".format(facetfitted[i]))
                        except:
                            if facetfitted[(i-1) % len(facetfitted)] is not None:
                                facetfitted[i] = facetfitted[(i-1) % len(facetfitted)]
                            elif facetfitted[(i+1) % len(facetfitted)] is not None:
                                facetfitted[i] = facetfitted[(i+1) % len(facetfitted)]
                            else:
                                continue
                    elif not(useCircles) and facetfitted[i] is None:
                        facetfitted[i] = notUseCircles_linear[i]

                #Make facets gapless
                if makeGapless:
                    for i in range(len(facetfitted)):
                        if facetfitted[i] is not None and facetfitted[i][0] is not 'corner' and facetfitted[i][0] is not 'curvedcorner' and facetfitted[(i+1) % len(facetfitted)] is not None and facetfitted[(i+1) % len(facetfitted)][0] is not 'corner' and facetfitted[(i+1) % len(facetfitted)][0] is not 'curvedcorner':
                            #average rightmost intersect of current facet and leftmost intersect of previous facet
                            facetpoint1 = facetfitted[i][-1][-1]
                            facetpoint2 = facetfitted[(i+1) % len(facetfitted)][-1][0]
                            newIntersect = lerp(facetpoint1, facetpoint2, 0.5)
                            facetfitted[i][-1][-1] = newIntersect
                            facetfitted[(i+1)% len(facetfitted)][-1][0] = newIntersect
                    for i in range(len(facetfitted)):
                        #update circular facet center if needed
                        if facetfitted[i] is not None and facetfitted[i][0] == 'arc':
                            intersectleft = facetfitted[i][3][0]
                            intersectright = facetfitted[i][3][-1]
                            if facetfitted[i][2] > 0:
                                facetfitted[i][1] = getCenter(intersectleft, intersectright, facetfitted[i][2])
                            else:
                                facetfitted[i][1] = getCenter(intersectright, intersectleft, -facetfitted[i][2])
                            #compute new center
                            #TODO: instead of preserving curvature, try preserving volume fraction
                
                #Store facet info for each grid square (from merged squares)
                for i in range(len(facetfitted)):
                    for facetsquares in mergedpolyinfos[path[i+1]][0]:
                        predfacets[facetsquares[0]][facetsquares[1]] = facetfitted[i]

                #Get unique facets
                for facet in facetfitted:
                    if facet not in facetsunique:
                        facetsunique.append(facet)

    #print(predfacets)

    return predfacets, facetsunique

def advectFacets(opolys, areas, predfacets, velocity, checksize, threshold):
    #areas after new timestep
    totalunshiftedareas = []

    for x in range(len(opolys)):
        for y in range(len(opolys)):
            if predfacets[x][y] is None and areas[x][y] > 0:
                totalunshiftedareas.append(areas[x][y]*getArea(opolys[x][y]))
            elif predfacets[x][y] is not None:
                predfacettype = predfacets[x][y][0]
                if predfacettype == 'linear':
                    totalunshiftedareas.append(getPolyLineArea(opolys[x][y], predfacets[x][y][1][0], predfacets[x][y][1][1]))
                elif predfacettype == 'corner':
                    cornerfacet = predfacets[x][y][1]
                    polyintersectioncornerarea = getPolyCornerArea(opolys[x][y], cornerfacet[0], cornerfacet[1], cornerfacet[2])
                    totalunshiftedareas.append(abs(polyintersectioncornerarea))
                elif predfacettype == 'arc':
                    if predfacets[x][y][2] > 0:
                        #convex facet
                        polyintersectioncirclearea, _ = getCircleIntersectArea([predfacets[x][y][1][0], predfacets[x][y][1][1]], predfacets[x][y][2], opolys[x][y])
                    else:
                        #concave facet
                        polyintersectioncirclearea, _ = getCircleIntersectArea([predfacets[x][y][1][0], predfacets[x][y][1][1]], -predfacets[x][y][2], opolys[x][y])
                        polyintersectioncirclearea = getArea(opolys[x][y]) - polyintersectioncirclearea
                    totalunshiftedareas.append(abs(polyintersectioncirclearea))
                else: #curved corner:
                    print("Curved corner: {}".format([x, y]))
                    curvedcorner = predfacets[x][y][-1]
                    polyintersectioncurvedcornerarea = getPolyCurvedCornerArea(opolys[x][y], curvedcorner[0], curvedcorner[1], curvedcorner[2], predfacets[x][y][3], predfacets[x][y][4])
                    totalunshiftedareas.append(abs(polyintersectioncurvedcornerarea))

    print("Preshifted area of facets: {}".format(sum(totalunshiftedareas)))

    totalgivenarea = 0
    for x in range(len(opolys)):
        for y in range(len(opolys)):
            totalgivenarea += areas[x][y]*getArea(opolys[x][y])
    print("Given area before facets: {}".format(totalgivenarea))

    totalshiftedareas = []

    nareas = [[[] for _ in range(len(opolys))] for _ in range(len(opolys))]
    newareas = [[None] * (len(opolys)) for _ in range(len(opolys))]
    
    shiftedfacets = []

    print("Shifting facets")

    for x in range(len(opolys)):
        for y in range(len(opolys)):
            if predfacets[x][y] is not None:
                #Shift facet based on velocity vector field:
                #For linear, corner, shift endpoints based on vector field
                #For arc, curvedcorner, shift endpoints and keep radii constant (fit centers as needed)
                predfacettype = predfacets[x][y][0]
                #print(predfacettype)
                if predfacettype == 'linear':
                    #Advect by using 3 control points: 2 endpoints + midpoint, turns into circular facet
                    preshiftcontrol1 = predfacets[x][y][-1][0]
                    preshiftcontrol2 = predfacets[x][y][-1][-1]
                    preshiftcontrol3 = lerp(preshiftcontrol1, preshiftcontrol2, 0.5)
                    shiftcontrol1 = [preshiftcontrol1[0]+velocity(preshiftcontrol1)[0], preshiftcontrol1[1]+velocity(preshiftcontrol1)[1]]
                    shiftcontrol2 = [preshiftcontrol2[0]+velocity(preshiftcontrol2)[0], preshiftcontrol2[1]+velocity(preshiftcontrol2)[1]]
                    shiftmid3 = [preshiftcontrol3[0]+velocity(preshiftcontrol3)[0], preshiftcontrol3[1]+velocity(preshiftcontrol3)[1]]
                    [shiftcirclecenter, shiftcircleradius] = getCircumcircle(shiftcontrol1, shiftcontrol2, shiftmid3)
                    if shiftcircleradius is None or shiftcircleradius > 1e3:
                        #Shifted facet is collinear, advect by creating linear facet
                        shiftlinearfacet = list(map(lambda x : [x[0]+velocity(x)[0], x[1]+velocity(x)[1]], predfacets[x][y][-1]))
                        shiftedfacets.append([predfacettype, shiftlinearfacet])
                    else:
                        #Shifted facet is not collinear, advect by creating arc facet
                        #Determine which side of line shiftcontrol1 to shiftcontrol2 the circumcenter lies
                        if (shiftcirclecenter[0]-shiftcontrol1[0])*(-(shiftcontrol2[1]-shiftcontrol1[1])) + (shiftcirclecenter[1]-shiftcontrol1[1])*(shiftcontrol2[0]-shiftcontrol1[0]) < 0:
                            #Circumcenter on right of line, need to invert radius
                            shiftcircleradius *= -1
                        predfacettype = 'arc'
                        shiftlinearfacet = [shiftcontrol1, shiftcontrol2]
                        shiftedfacets.append([predfacettype, shiftcirclecenter, shiftcircleradius, shiftlinearfacet])
                    
                elif predfacettype == 'corner':
                    #Advect by using 3 control points: 2 endpoints + midpoint per side, turns into curvedcorner facet
                    preshiftcorner1 = predfacets[x][y][-1][0]
                    preshiftcorner2 = predfacets[x][y][-1][1]
                    preshiftmid1 = lerp(preshiftcorner1, preshiftcorner2, 0.5)
                    shiftcorner1 = [preshiftcorner1[0]+velocity(preshiftcorner1)[0], preshiftcorner1[1]+velocity(preshiftcorner1)[1]]
                    shiftcorner2 = [preshiftcorner2[0]+velocity(preshiftcorner2)[0], preshiftcorner2[1]+velocity(preshiftcorner2)[1]]
                    shiftmid1 = [preshiftmid1[0]+velocity(preshiftmid1)[0], preshiftmid1[1]+velocity(preshiftmid1)[1]]
                    [shiftcirclecenter1, shiftcircleradius1] = getCircumcircle(shiftcorner1, shiftcorner2, shiftmid1)
                    if shiftcircleradius1 is None or shiftcircleradius1 > 1e3:
                        shiftcirclecenter1 = None
                        shiftcircleradius1 = None
                    elif shiftcircleradius1 is not None and (shiftcirclecenter1[0]-shiftcorner1[0])*(-(shiftcorner2[1]-shiftcorner1[1])) + (shiftcirclecenter1[1]-shiftcorner1[1])*(shiftcorner2[0]-shiftcorner1[0]) < 0:
                        shiftcircleradius1 *= -1

                    preshiftcorner3 = predfacets[x][y][-1][2]
                    preshiftmid2 = lerp(preshiftcorner2, preshiftcorner3, 0.5)
                    shiftcorner3 = [preshiftcorner3[0]+velocity(preshiftcorner3)[0], preshiftcorner3[1]+velocity(preshiftcorner3)[1]]
                    shiftmid2 = [preshiftmid2[0]+velocity(preshiftmid2)[0], preshiftmid2[1]+velocity(preshiftmid2)[1]]
                    [shiftcirclecenter2, shiftcircleradius2] = getCircumcircle(shiftcorner2, shiftcorner3, shiftmid2)
                    if shiftcircleradius2 is None or shiftcircleradius2 > 1e3:
                        shiftcirclecenter2 = None
                        shiftcircleradius2 = None
                    elif shiftcircleradius2 is not None and (shiftcirclecenter2[0]-shiftcorner2[0])*(-(shiftcorner3[1]-shiftcorner2[1])) + (shiftcirclecenter2[1]-shiftcorner2[1])*(shiftcorner3[0]-shiftcorner2[0]) < 0:
                        shiftcircleradius2 *= -1

                    predfacettype = 'curvedcorner'
                    shiftcurvedcorner = [shiftcorner1, shiftcorner2, shiftcorner3]
                    print("Corner: {}, {}".format(shiftcircleradius1, shiftcircleradius2))
                    shiftedfacets.append([predfacettype, shiftcirclecenter1, shiftcirclecenter2, shiftcircleradius1, shiftcircleradius2, shiftcurvedcorner])

                elif predfacettype == 'curvedcorner':
                    #Advect by using 3 control points: 2 endpoints + midpoint per side, turns into curvedcorner facet
                    if predfacets[x][y][3] is None:
                        #Left side is linear:
                        preshiftcorner1 = predfacets[x][y][-1][0]
                        preshiftcorner2 = predfacets[x][y][-1][1]
                        preshiftmid1 = lerp(preshiftcorner1, preshiftcorner2, 0.5)
                        shiftcorner1 = [preshiftcorner1[0]+velocity(preshiftcorner1)[0], preshiftcorner1[1]+velocity(preshiftcorner1)[1]]
                        shiftcorner2 = [preshiftcorner2[0]+velocity(preshiftcorner2)[0], preshiftcorner2[1]+velocity(preshiftcorner2)[1]]
                        shiftmid1 = [preshiftmid1[0]+velocity(preshiftmid1)[0], preshiftmid1[1]+velocity(preshiftmid1)[1]]
                        [shiftcirclecenter1, shiftcircleradius1] = getCircumcircle(shiftcorner1, shiftcorner2, shiftmid1)
                        if shiftcircleradius1 is None or shiftcircleradius1 > 1e3:
                            shiftcirclecenter1 = None
                            shiftcircleradius1 = None
                        elif shiftcircleradius1 is not None and (shiftcirclecenter1[0]-shiftcorner1[0])*(-(shiftcorner2[1]-shiftcorner1[1])) + (shiftcirclecenter1[1]-shiftcorner1[1])*(shiftcorner2[0]-shiftcorner1[0]) < 0:
                            shiftcircleradius1 *= -1
                    else:
                        #Left side is arc
                        preshiftcorner1 = predfacets[x][y][-1][0]
                        preshiftcorner2 = predfacets[x][y][-1][1]
                        preshiftmid1 = lerp(preshiftcorner1, preshiftcorner2, 0.5)
                        preshiftcirclecenter1 = predfacets[x][y][1]
                        preshiftcircleradius1 = predfacets[x][y][3]
                        preshiftconstant = getDistance(preshiftmid1, [0, 0])
                        preshiftmid1 = [preshiftcirclecenter1[0] + preshiftcircleradius1/preshiftconstant*preshiftmid1[0], preshiftcirclecenter1[1] + preshiftcircleradius1/preshiftconstant*preshiftmid1[1]]
                        shiftcorner1 = [preshiftcorner1[0]+velocity(preshiftcorner1)[0], preshiftcorner1[1]+velocity(preshiftcorner1)[1]]
                        shiftcorner2 = [preshiftcorner2[0]+velocity(preshiftcorner2)[0], preshiftcorner2[1]+velocity(preshiftcorner2)[1]]
                        shiftmid1 = [preshiftmid1[0]+velocity(preshiftmid1)[0], preshiftmid1[1]+velocity(preshiftmid1)[1]]
                        [shiftcirclecenter1, shiftcircleradius1] = getCircumcircle(shiftcorner1, shiftcorner2, shiftmid1)
                        if shiftcircleradius1 is not None and (shiftcirclecenter1[0]-shiftcorner1[0])*(-(shiftcorner2[1]-shiftcorner1[1])) + (shiftcirclecenter1[1]-shiftcorner1[1])*(shiftcorner2[0]-shiftcorner1[0]) < 0:
                            shiftcircleradius1 *= -1

                    if predfacets[x][y][4] is None:
                        #Right side is linear
                        preshiftcorner3 = predfacets[x][y][-1][2]
                        preshiftmid2 = lerp(preshiftcorner2, preshiftcorner3, 0.5)
                        shiftcorner3 = [preshiftcorner3[0]+velocity(preshiftcorner3)[0], preshiftcorner3[1]+velocity(preshiftcorner3)[1]]
                        shiftmid2 = [preshiftmid2[0]+velocity(preshiftmid2)[0], preshiftmid2[1]+velocity(preshiftmid2)[1]]
                        [shiftcirclecenter2, shiftcircleradius2] = getCircumcircle(shiftcorner2, shiftcorner3, shiftmid2)
                        if shiftcircleradius2 is None or shiftcircleradius2 > 1e3:
                            shiftcirclecenter2 = None
                            shiftcircleradius2 = None
                        elif shiftcircleradius2 is not None and (shiftcirclecenter2[0]-shiftcorner2[0])*(-(shiftcorner3[1]-shiftcorner2[1])) + (shiftcirclecenter2[1]-shiftcorner2[1])*(shiftcorner3[0]-shiftcorner2[0]) < 0:
                            shiftcircleradius2 *= -1
                    else:
                        #Right side is arc
                        preshiftcorner3 = predfacets[x][y][-1][2]
                        preshiftmid2 = lerp(preshiftcorner2, preshiftcorner3, 0.5)
                        preshiftcirclecenter2 = predfacets[x][y][2]
                        preshiftcircleradius2 = predfacets[x][y][4]
                        preshiftconstant = getDistance(preshiftmid2, [0, 0])
                        preshiftmid2 = [preshiftcirclecenter2[0] + preshiftcircleradius2/preshiftconstant*preshiftmid2[0], preshiftcirclecenter2[1] + preshiftcircleradius2/preshiftconstant*preshiftmid2[1]]
                        shiftcorner3 = [preshiftcorner3[0]+velocity(preshiftcorner3)[0], preshiftcorner3[1]+velocity(preshiftcorner3)[1]]
                        shiftmid2 = [preshiftmid2[0]+velocity(preshiftmid2)[0], preshiftmid2[1]+velocity(preshiftmid2)[1]]
                        [shiftcirclecenter2, shiftcircleradius2] = getCircumcircle(shiftcorner2, shiftcorner3, shiftmid2)
                        if shiftcircleradius2 is not None and (shiftcirclecenter2[0]-shiftcorner2[0])*(-(shiftcorner3[1]-shiftcorner2[1])) + (shiftcirclecenter2[1]-shiftcorner2[1])*(shiftcorner3[0]-shiftcorner2[0]) < 0:
                            shiftcircleradius2 *= -1

                    predfacettype = 'curvedcorner'
                    shiftcurvedcorner = [shiftcorner1, shiftcorner2, shiftcorner3]
                    print("Curved corner: {}, {}".format(shiftcircleradius1, shiftcircleradius2))
                    shiftedfacets.append([predfacettype, shiftcirclecenter1, shiftcirclecenter2, shiftcircleradius1, shiftcircleradius2, shiftcurvedcorner])

                else:
                    #Circular facets
                    #Advect by using 3 control points: 2 endpoints + midpoint projected onto circular facet, turns into circular facet with different curvature
                    preshiftcontrol1 = predfacets[x][y][-1][0]
                    preshiftcontrol2 = predfacets[x][y][-1][-1]
                    preshiftcontrol3 = lerp(preshiftcontrol1, preshiftcontrol2, 0.5)
                    preshiftcircleradius = abs(predfacets[x][y][2])
                    preshiftcirclecenter = predfacets[x][y][1]
                    #if circle is small:
                    if ((preshiftcirclecenter[0]-preshiftcontrol1[0])*(preshiftcontrol3[1]-preshiftcontrol1[1])-(preshiftcirclecenter[1]-preshiftcontrol1[1])*(preshiftcontrol3[0]-preshiftcontrol1[0]))*predfacets[x][y][2] > 0:
                        preshiftcontrol3 = lerp(preshiftcontrol3, preshiftcirclecenter, 2)
                    preshiftcontrol3 = [preshiftcontrol3[0]-preshiftcirclecenter[0], preshiftcontrol3[1]-preshiftcirclecenter[1]]
                    preshiftconstant = getDistance(preshiftcontrol3, [0, 0])
                    preshiftcontrol3 = [preshiftcirclecenter[0] + preshiftcircleradius/preshiftconstant*preshiftcontrol3[0], preshiftcirclecenter[1] + preshiftcircleradius/preshiftconstant*preshiftcontrol3[1]]

                    shiftcontrol1 = [preshiftcontrol1[0]+velocity(preshiftcontrol1)[0], preshiftcontrol1[1]+velocity(preshiftcontrol1)[1]]
                    shiftcontrol2 = [preshiftcontrol2[0]+velocity(preshiftcontrol2)[0], preshiftcontrol2[1]+velocity(preshiftcontrol2)[1]]
                    shiftcontrol3 = [preshiftcontrol3[0]+velocity(preshiftcontrol3)[0], preshiftcontrol3[1]+velocity(preshiftcontrol3)[1]]
                    [shiftcirclecenter, shiftcircleradius] = getCircumcircle(shiftcontrol1, shiftcontrol2, shiftcontrol3)
                    if shiftcircleradius is not None and (shiftcontrol3[0]-shiftcontrol1[0])*(-(shiftcontrol2[1]-shiftcontrol1[1])) + (shiftcontrol3[1]-shiftcontrol1[1])*(shiftcontrol2[0]-shiftcontrol1[0]) > 0:
                        shiftcircleradius *= -1
                    #print("Curvature original: {}, shifted: {}, difference: {}".format(1/shiftcircleradius, 1/predfacets[x][y][2], 1/shiftcircleradius-1/predfacets[x][y][2]))
                    if shiftcircleradius is None:
                        #Shifted facet is collinear, advect by maintaining curvature
                        shiftcircleintersects = list(map(lambda x : [x[0]+velocity(x)[0], x[1]+velocity(x)[1]], predfacets[x][y][-1]))
                        shiftcircleradius = predfacets[x][y][2]
                        shiftcirclecenter = getCenter(shiftcircleintersects[0], shiftcircleintersects[-1], shiftcircleradius)
                        shiftedfacets.append([predfacettype, shiftcirclecenter, shiftcircleradius, shiftcircleintersects])
                    else:
                        #Shifted facet is not collinear
                        shiftedfacets.append([predfacettype, shiftcirclecenter, shiftcircleradius, [shiftcontrol1, shiftcontrol2]])

                #print(shiftedfacets[-1])

                #TODO: fix polygon intersection algorithm, makeshift solution: shift by 1e-15
                trueshiftpoly = list(map(lambda x : [x[0]+velocity(x)[0], x[1]+velocity(x)[1]], opolys[x][y]))
                shiftpoly = list(map(lambda x : [x[0]+velocity(x)[0], x[1]+velocity(x)[1]], opolys[x][y]))

                shiftbounds = [min(list(map(lambda x : x[0], shiftpoly))), min(list(map(lambda x : x[1], shiftpoly))), max(list(map(lambda x : x[0], shiftpoly))), max(list(map(lambda x : x[1], shiftpoly)))]
                advectedarea = 0
                #bounds: minx, miny, maxx, maxy
                #For each neighbor of the cell
                for testx in range(-checksize, checksize+1):
                    for testy in range(-checksize, checksize+1):
                        checkx = x-testx
                        checky = y-testy
                        if checkx >= 0 and checkx < len(opolys) and checky >= 0 and checky < len(opolys[0]):
                            testpoly = opolys[checkx][checky]
                            testbounds = [min(list(map(lambda x : x[0], testpoly))), min(list(map(lambda x : x[1], testpoly))), max(list(map(lambda x : x[0], testpoly))), max(list(map(lambda x : x[1], testpoly)))]
                            if not(testbounds[2] <= shiftbounds[0] or shiftbounds[2] <= testbounds[0] or testbounds[3] <= shiftbounds[1] or shiftbounds[3] <= testbounds[1]):
                                #bounding boxes intersect, could be nonzero intersection
                                try:
                                    polyintersections = getPolyIntersectArea(shiftpoly, testpoly)
                                except:
                                    print("Failed polyintersect: getPolyIntersectArea({}, {})".format(shiftpoly, testpoly))
                                    shiftpoly = list(map(lambda x : [x[0]+1e-13, x[1]+1e-13], shiftpoly))
                                    polyintersections = getPolyIntersectArea(shiftpoly, testpoly)
                                if len(polyintersections) == 0:
                                    #No intersection
                                    continue
                                #For each overlap region
                                for polyintersection in polyintersections:
                                    if predfacettype == 'linear':
                                        polyintersectionlineararea = getPolyLineArea(polyintersection, shiftlinearfacet[0], shiftlinearfacet[len(shiftlinearfacet)-1])
                                        nareas[checkx][checky].append([abs(polyintersectionlineararea), [x, y]])
                                        advectedarea += abs(polyintersectionlineararea)
                                    #elif predfacettype == 'corner':
                                    #    polyintersectioncornerarea = getPolyCornerArea(polyintersection, shiftcornerfacet[0], shiftcornerfacet[1], shiftcornerfacet[2])
                                    #    nareas[checkx][checky].append([abs(polyintersectioncornerarea), [x, y]])
                                    elif predfacettype == 'arc':
                                        if shiftcircleradius > 0:
                                            polyintersectioncirclearea, _ = getCircleIntersectArea(shiftcirclecenter, shiftcircleradius, polyintersection)
                                        else:
                                            polyintersectioncirclearea, _ = getCircleIntersectArea(shiftcirclecenter, -shiftcircleradius, polyintersection)
                                            polyintersectioncirclearea = getArea(polyintersection) - polyintersectioncirclearea
                                        nareas[checkx][checky].append([abs(polyintersectioncirclearea), [x, y]])
                                        advectedarea += abs(polyintersectioncirclearea)
                                    elif predfacettype == 'curvedcorner':
                                        polyintersectioncurvedcornerarea = getPolyCurvedCornerArea(polyintersection, shiftcurvedcorner[0], shiftcurvedcorner[1], shiftcurvedcorner[2], shiftcircleradius1, shiftcircleradius2)
                                        nareas[checkx][checky].append([abs(polyintersectioncurvedcornerarea), [x, y]])
                                        advectedarea += abs(polyintersectioncurvedcornerarea)
                #Area is not conserved: issue
                if abs(areas[x][y]*getArea(opolys[x][y])-advectedarea) > 1e-4:
                    print("Square: {}, Facet: {}, Partial original area: {}, Advected: {}, Error: {}".format([x, y], predfacets[x][y], areas[x][y]*getArea(opolys[x][y]), advectedarea, areas[x][y]*getArea(opolys[x][y])-advectedarea))
                    print(opolys[x][y])
                    print("Shifted opoly: {}".format(list(map(lambda x : [x[0]+velocity(x)[0], x[1]+velocity(x)[1]], opolys[x][y]))))
                    print("Shifted facet: {}".format(shiftedfacets[-1]))
                    for testx in range(-checksize, checksize+1):
                        for testy in range(-checksize, checksize+1):
                            checkx = x-testx
                            checky = y-testy
                            if checkx >= 0 and checkx < len(opolys) and checky >= 0 and checky < len(opolys[0]):
                                testpoly = opolys[checkx][checky]
                                testbounds = [min(list(map(lambda x : x[0], testpoly))), min(list(map(lambda x : x[1], testpoly))), max(list(map(lambda x : x[0], testpoly))), max(list(map(lambda x : x[1], testpoly)))]
                                if not(testbounds[2] <= shiftbounds[0] or shiftbounds[2] <= testbounds[0] or testbounds[3] <= shiftbounds[1] or shiftbounds[3] <= testbounds[1]):
                                    #bounding boxes intersect, could be nonzero intersection
                                    try:
                                        polyintersections = getPolyIntersectArea(shiftpoly, testpoly)
                                    except:
                                        print("Failed polyintersect: getPolyIntersectArea({}, {})".format(shiftpoly, testpoly))
                                        shiftpoly = list(map(lambda x : [x[0]+1e-13, x[1]+1e-13], shiftpoly))
                                        polyintersections = getPolyIntersectArea(shiftpoly, testpoly)
                                    if len(polyintersections) == 0:
                                        #No intersection
                                        continue
                                    #For each overlap region
                                    for polyintersection in polyintersections:
                                        print("Neighbor: {}, polyintersection: {}".format(opolys[checkx][checky], polyintersection))
                                        if predfacettype == 'linear':
                                            polyintersectionlineararea = getPolyLineArea(polyintersection, shiftlinearfacet[0], shiftlinearfacet[len(shiftlinearfacet)-1])
                                            print(abs(polyintersectionlineararea))
                                        #elif predfacettype == 'corner':
                                        #    polyintersectioncornerarea = getPolyCornerArea(polyintersection, shiftcornerfacet[0], shiftcornerfacet[1], shiftcornerfacet[2])
                                        #    nareas[checkx][checky].append([abs(polyintersectioncornerarea), [x, y]])
                                        elif predfacettype == 'arc':
                                            if shiftcircleradius > 0:
                                                polyintersectioncirclearea, _ = getCircleIntersectArea(shiftcirclecenter, shiftcircleradius, polyintersection)
                                            else:
                                                polyintersectioncirclearea, _ = getCircleIntersectArea(shiftcirclecenter, -shiftcircleradius, polyintersection)
                                                polyintersectioncirclearea = getArea(polyintersection) - polyintersectioncirclearea
                                            print(abs(polyintersectioncirclearea))
                                        elif predfacettype == 'curvedcorner':
                                            polyintersectioncurvedcornerarea = getPolyCurvedCornerArea(polyintersection, shiftcurvedcorner[0], shiftcurvedcorner[1], shiftcurvedcorner[2], shiftcircleradius1, shiftcircleradius2)
                                            print(abs(polyintersectioncurvedcornerarea))
                                    

            elif areas[x][y] == 1:
                """
                neighbors_full = True
                for testx in range(-checksize, checksize+1):
                    for testy in range(-checksize, checksize+1):
                        checkx = x-testx
                        checky = y-testy
                        if checkx >= 0 and checkx < len(opolys) and checky >= 0 and checky < len(opolys[0]) and (predfacets[checkx][checky] is not None or areas[checkx][checky] < 1):
                            neighbors_full = False
                if neighbors_full:
                    #All neighbors are full, so just fill [x, y] and continue
                    newareas[x][y] = abs(getArea(opolys[x][y]))
                    continue
                #Otherwise compute polygon intersections
                """

                #TODO: fix polygon intersection algorithm, makeshift solution: shift by 1e-15
                trueshiftpoly = list(map(lambda x : [x[0]+velocity(x)[0], x[1]+velocity(x)[1]], opolys[x][y]))
                shiftpoly = list(map(lambda x : [x[0]+velocity(x)[0], x[1]+velocity(x)[1]], opolys[x][y]))

                shiftbounds = [min(list(map(lambda x : x[0], shiftpoly))), min(list(map(lambda x : x[1], shiftpoly))), max(list(map(lambda x : x[0], shiftpoly))), max(list(map(lambda x : x[1], shiftpoly)))]
                advectedarea = 0
                #bounds: minx, miny, maxx, maxy
                for testx in range(-checksize, checksize+1):
                    for testy in range(-checksize, checksize+1):
                        checkx = x-testx
                        checky = y-testy
                        if checkx >= 0 and checkx < len(opolys) and checky >= 0 and checky < len(opolys[0]):
                            testpoly = opolys[checkx][checky]
                            testbounds = [min(list(map(lambda x : x[0], testpoly))), min(list(map(lambda x : x[1], testpoly))), max(list(map(lambda x : x[0], testpoly))), max(list(map(lambda x : x[1], testpoly)))]
                            if not(testbounds[2] <= shiftbounds[0] or shiftbounds[2] <= testbounds[0] or testbounds[3] <= shiftbounds[1] or shiftbounds[3] <= testbounds[1]):
                                #bounding boxes intersect, could be nonzero intersection
                                try:
                                    #print("Doing: {}".format(testpoly))
                                    polyintersections = getPolyIntersectArea(shiftpoly, testpoly)
                                except:
                                    print("Failed polyintersect: getPolyIntersectArea({}, {})".format(shiftpoly, testpoly))
                                    shiftpoly = list(map(lambda x : [x[0]+1e-13, x[1]+1e-13], shiftpoly))
                                    polyintersections = getPolyIntersectArea(shiftpoly, testpoly)
                                if len(polyintersections) == 0:
                                    #No intersection
                                    continue
                                #For each overlap region
                                for polyintersection in polyintersections:
                                    nareas[checkx][checky].append([abs(getArea(polyintersection)), [x, y]])
                                    advectedarea += abs(getArea(polyintersection))
                #Area is not conserved: issue
                if abs(areas[x][y]*getArea(opolys[x][y])-advectedarea) > 1e-4:
                    print("Square: {}, Full original area: {}, Advected: {}, Error: {}".format([x, y], areas[x][y]*getArea(opolys[x][y]), advectedarea, areas[x][y]*getArea(opolys[x][y])-advectedarea))

            #Unpredicted partial fraction
            elif areas[x][y] > 0:
                print("Unpredicted partial fraction: {}, {}".format(x, y))

    #Plot shifted facets
    plotFacets(shiftedfacets, 'shiftedfacets')

    #Sum immediate advected areas, could have issues
    for x in range(len(opolys)):
        for y in range(len(opolys)):
            if newareas[x][y] is None:
                newareas[x][y] = sum(list(map(lambda z : z[0], nareas[x][y])))

    simpleadvectedarea = sum(list(map(lambda z : sum(z), newareas)))
    print("Sum after simple advection: {}".format(simpleadvectedarea))

    #Global volume conservation
    #Loop through and fix all cells with volume close enough to full or empty
    sumofexcesses = 0 #This much volume needs to be readded to account for the adjustments
    summixedvolumes = 0
    for x in range(len(opolys)):
        for y in range(len(opolys)):
            if newareas[x][y] > getArea(opolys[x][y]): #Cell with excess volume
                print("Excess volume cell: {}, {}".format([x, y], newareas[x][y]/getArea(opolys[x][y])))
                sumofexcesses += newareas[x][y]-getArea(opolys[x][y])
                if newareas[x][y]/getArea(opolys[x][y]) > 1.01:
                    print("A lot of excess: {}, {}, {}".format(x, y, nareas[x][y]))
                newareas[x][y] = getArea(opolys[x][y])
            elif newareas[x][y] != 0 and abs(newareas[x][y])/getArea(opolys[x][y]) < threshold: #Cell close to empty
                print("Empty volume cell: {}, {}".format([x, y], newareas[x][y]/getArea(opolys[x][y])))
                sumofexcesses += newareas[x][y]
                newareas[x][y] = 0
            elif newareas[x][y] != getArea(opolys[x][y]) and abs(1 - (newareas[x][y]/getArea(opolys[x][y]))) < threshold: #Cell close to full
                print("Full volume cell: {}, {}".format([x, y], newareas[x][y]/getArea(opolys[x][y])))
                sumofexcesses -= getArea(opolys[x][y])-newareas[x][y]
                newareas[x][y] = getArea(opolys[x][y])
            elif newareas[x][y] != getArea(opolys[x][y]):
                summixedvolumes += newareas[x][y]

    print("Sum of excesses: {}".format(sumofexcesses))

    #Tries to ensure global volume conservation by scaling volumes of all mixed cells (done proportionally by volume)
    globaladjustment = totalgivenarea-simpleadvectedarea-sumofexcesses #This much volume needs to be readded to all partial squares
    fixexcessfraction = 1 + globaladjustment/summixedvolumes

    print("Sum of mixed volumes: {}".format(summixedvolumes))

    print("fixexcessfraction: {}".format(fixexcessfraction))
    fixexcessfraction = 1
    assert fixexcessfraction > 0, "Amount of error during advection is too large, global volume conservation failed!"

    for x in range(len(opolys)):
        for y in range(len(opolys)):
            if threshold/2 < newareas[x][y]/getArea(opolys[x][y]) and newareas[x][y]/getArea(opolys[x][y]) < 1-threshold/2:
                if newareas[x][y]*fixexcessfraction/getArea(opolys[x][y]) < 1-threshold and newareas[x][y]*fixexcessfraction/getArea(opolys[x][y]) > threshold:
                    newareas[x][y] = newareas[x][y]*fixexcessfraction
                if newareas[x][y]*fixexcessfraction > getArea(opolys[x][y]):
                    newareas[x][y] = getArea(opolys[x][y])
                    print("Lost {} of volume: {}".format(newareas[x][y]*fixexcessfraction - newareas[x][y], [x, y]))
    
    print("Final sum of volumes: {}".format(sum(list(map(lambda x : sum(x), newareas)))))

    for x in range(len(opolys)):
        for y in range(len(opolys)):
            newareas[x][y] /= getArea(opolys[x][y])

    #print(newareas)

    return newareas