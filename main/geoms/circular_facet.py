import math
import numpy as np
from main.geoms.geoms import getArea, getDistance, lerp, getPolyLineArea, getPolyLineIntersects, getCentroid, pointInPoly, pointRightOfLine, pointLeftOfLine
from main.geoms.linear_facet import getLinearFacet, getLinearFacetFromNormal
from scipy.optimize import minimize, root, newton

#True if major arc
def isMajorArc(pLeft, pRight, center, radius):
    return pointLeftOfLine(pRight, center, pLeft) == (radius < 0)

#Assume points are actually on circle
#Two checks: if negative radius, return negative area; if major arc, return larger area
def getArcArea(p1, p2, center, radius):
    #Compute area
    distance = getDistance(p1, p2)
    theta = math.asin(distance/(2*abs(radius)))
    area = (theta * radius**2) - (distance * math.cos(theta)*abs(radius) / 2)
    #Check orientation of segment p1p2 compared to center
    if pointLeftOfLine(p2, p1, center):
        #Angle from p1 to p2 moves counterclockwise: return the small area
        area = -area + math.pi*radius**2
    #Check if negative curvature
    if radius < 0:
        #Returned area should be negative
        area -= math.pi*radius**2

    return area

#Newton's method never exceeds maxRadius. TODO: can add code verifying that maxRadius is valid upper bound
def matchArcArea(distance, area, epsilon, maxRadius=1e6):
    #print("Running matchArcArea({}, {}, {})".format(distance, area, epsilon))
    if area == 0:
        #Zero curvature, hopefully should never be called
        #print("Calling matchArcArea with area = 0")
        return float("inf")
    elif area < 0:
        return -1 * matchArcArea(distance, -area, epsilon)

    #Hyperparameters:
    dtbase = 1e-6 #Timestep used to calculate numerical estimates of derivatives
    dtscale = 10 #Scale by this much per iteration
    maxIterations = 50

    if abs(area - distance**2 / 8 * math.pi) < epsilon:
        return distance/2

    if area > distance**2 / 8 * math.pi:
        #big arc area
        def arc_area(radius):
            if radius > distance/2:
                return ((math.pi - math.asin(distance/(2*abs(radius)))) * radius**2) + (distance * math.cos(math.asin(distance/(2*abs(radius))))*abs(radius)/2)
            #else:
            #    return ((math.asin(2*abs(radius)/distance)) * (distance**2/(4*abs(radius)))**2) - (distance * math.cos(math.asin(2*abs(radius)/distance))*(distance**2/(4*abs(radius)))/2)
        def arc_deriv(radius):
            if radius > distance/2:
                return (2*radius*(math.pi - math.asin(distance/(2*abs(radius))))) + (2*distance*radius/math.sqrt(4*radius**2 - distance**2))
            #else:
            #    return
        arc_area = lambda radius : ((math.pi - math.asin(distance/(2*abs(radius)))) * radius**2) + (distance * math.cos(math.asin(distance/(2*abs(radius))))*abs(radius)/2)
        arc_deriv = lambda radius : (2*radius*(math.pi - math.asin(distance/(2*abs(radius))))) + (2*distance*radius/math.sqrt(4*radius**2 - distance**2))
        #test for maximum allowed radius
        if area > arc_area(maxRadius):
            return float("inf")
    else:
        #small arc area
        def arc_area(radius):
            if radius > distance/2:
                return (math.asin(distance/(2*abs(radius))) * radius**2) - (distance * math.cos(math.asin(distance/(2*abs(radius))))*abs(radius)/2)
            #else:
            #    return ((math.pi - math.asin(2*abs(radius))/distance) * radius**2) + (distance * math.cos(math.asin(2*abs(radius))/distance)*(distance**2/(4*abs(radius)))/2)
        def arc_deriv(radius):
            if radius > distance/2:
                return (2*radius*math.asin(distance/(2*radius))) - (2*distance*radius/math.sqrt(4*radius**2 - distance**2))
            #else:
            #    return
        arc_area = lambda radius : (math.asin(distance/(2*abs(radius))) * radius**2) - (distance * math.cos(math.asin(distance/(2*abs(radius))))*abs(radius)/2)
        arc_deriv = lambda radius : (2*radius*math.asin(distance/(2*radius))) - (2*distance*radius/math.sqrt(4*radius**2 - distance**2))
        #test for maximum allowed radius
        if area < arc_area(maxRadius):
            return float("inf")

    converged = False
    gr = distance/2
    prevgr = gr
    numIterations = 0
    while not(converged) and numIterations < maxIterations:
        cura = arc_area(gr)
        curaplusdt = cura
        if gr == distance/2:
            dadt = (arc_area(gr+dtbase)-arc_area(gr))/dtbase
        else:
            dadt = arc_deriv(gr)
        """
        dt = dtbase
        while curaplusdt - cura == 0:
            #Choose a good dt
            grplusdt = gr + dt
            curaplusdt = arc_area(grplusdt)
            cura = arc_area(gr)
            dadt = (curaplusdt - cura)/dt
            dt *= dtscale
        #print(dt)
        """

        if gr - ((cura-area)/dadt) >= distance/2:
            gr = min(gr - ((cura-area)/dadt), maxRadius)
        #gr must be >= distance/2
        else:
            gr = (gr+distance/2)/2

        if gr < 0:
            print(gr)
        #print(gr)

        numIterations += 1

        #print(arc_area(gr))
        if abs(arc_area(gr) - area) < epsilon:
            converged = True

    #print("Output: radius = {}".format(gr))
    #print("Final error: {}".format(arc_area(gr)-area))
    return gr

#Get center to the left of line l1 to l2
def getCenter(l1, l2, radius):
    if radius < 0:
        return getCenter(l2, l1, -radius)
    assert getDistance(l1, l2) <= 2*radius, "Error in getCenter arguments: {} > 2*{} (radius too small)".format(getDistance(l1, l2), radius)
    center = lerp(l1, l2, 0.5)
    c = math.sqrt(radius**2 - (getDistance(l1, l2)**2)/4)/getDistance(l1, l2)
    return [center[0]+(c*(l1[1]-l2[1])), center[1]+(c*(l2[0]-l1[0]))]

#Return [circumcenter, circumradius]
def getCircumcircle(p1, p2, p3):
    parallelthreshold = 1e-8 #Used to evaluate when determinant is too small, considered as collinear points. TODO: replace with something that considers relative distance of points as well?
    max_radius = 1e4 #Radius above this is considered straight line

    a = p2[1]-p1[1]
    b = p2[1]-p3[1]
    c = (p1[0]-p3[0])/2
    d = p2[0]-p1[0]
    e = p2[0]-p3[0]
    f = (p3[1]-p1[1])/2
    d12sq = a**2 + c**2
    d23sq = b**2 + e**2
    d13sq = 4*(c**2 + f**2)
    if (p1[0] == p2[0] and p2[0] == p3[0]) or abs(a*e-b*d)/max(d12sq, d23sq, d13sq) < parallelthreshold:
        #Treat as collinear points
        return [None, None]
    t = (a*f-c*d)/(a*e-b*d)
    circumcenter = [(p2[0]+p3[0])/2 + t*b, (p2[1]+p3[1])/2 - t*e]
    d1 = getDistance(circumcenter, p1)
    d2 = getDistance(circumcenter, p2)
    d3 = getDistance(circumcenter, p3)
    if max(abs(d1-d2), abs(d2-d3), abs(d3-d1)) > 1e-8:
        print("getCircumcircle({}, {}, {})".format(p1, p2, p3))
        print(d1)
        print(d2)
        print(d3)
    circumradius = (d1+d2+d3)/3
    if circumradius < max_radius:
        return [circumcenter, circumradius]
    else:
        return [None, None]

#Get circle-circle intersects
def getCircleCircleIntersects(center1, center2, radius1, radius2):
    d = getDistance(center1, center2)
    if abs(radius1-radius2) > d or radius1+radius2 < d or d == 0:
        return []
    else:
        chorddistance = (abs(radius1**2 - radius2**2)+d**2)/(2*d)
        halfchordlength = math.sqrt(radius1**2 - chorddistance**2)
        chordmidpoint = lerp(center1, center2, chorddistance/d)
        return [[chordmidpoint[0]+halfchordlength*(center2[1]-center1[1])/d, chordmidpoint[1]-halfchordlength*(center2[0]-center1[0])/d], [chordmidpoint[0]-halfchordlength*(center2[1]-center1[1])/d, chordmidpoint[1]+halfchordlength*(center2[0]-center1[0])/d]]

#Line l1 to l2
#If checkWithinLine, only returns intersects that lie within segment l1 to l2
def getCircleLineIntersects(l1, l2, center, radius, checkWithinLine=True):
    intersects = []
    a = (l2[0]-l1[0])**2 + (l2[1]-l1[1])**2
    b = 2*((l1[0]-center[0])*(l2[0]-l1[0]) + (l1[1]-center[1])*(l2[1]-l1[1]))
    c = (l1[0]-center[0])**2 + (l1[1]-center[1])**2 - radius**2
    #a is never 0 because p1 != p2
    disc = b**2 - 4*a*c
    if disc > 0:
        x1 = (-b - math.sqrt(disc))/(2*a)
        if (checkWithinLine and x1 <= 1 and x1 > 0) or not(checkWithinLine):
            inter1 = lerp(l1, l2, x1)
            intersects.append(inter1)
        x2 = (-b + math.sqrt(disc))/(2*a)
        if (checkWithinLine and x2 < 1 and x2 >= 0) or not(checkWithinLine):
            inter2 = lerp(l1, l2, x2)
            intersects.append(inter2)
    return intersects

def getCircleIntersectArea(center, radius, poly): #TODO fix
    #If radius is negative, reverse arcpoints
    if radius < 0:
        complement_area, arcpoints = getCircleIntersectArea(center, -radius, poly)
        arcpoints.reverse()
        return getArea(poly)-complement_area, arcpoints

    #Hyperparameter
    adjustcorneramount = 1e-14
    notmod = True
    while notmod:
        startAt = 1
        intersectpoints = []
        arcpoints = []
        for i in range(len(poly)):
            curpoint = poly[i]
            nextpoint = poly[(i+1) % len(poly)]
            curin = getDistance(curpoint, center) <= abs(radius)
            nextin = getDistance(nextpoint, center) <= abs(radius)
            if curin:
                intersectpoints.append(curpoint)
            lineintersects = getCircleLineIntersects(curpoint, nextpoint, center, abs(radius))
            for intersect in lineintersects:
                intersectpoints.append(intersect)
                if len(arcpoints) == 0 and curin and not(nextin):
                    startAt = 0
                arcpoints.append(intersect)
        #If not 0 mod 2, circle intersects a corner, perturb poly and rerun
        if len(arcpoints) % 2 == 1:
            poly = list(map(lambda x : [x[0]+adjustcorneramount, x[1]+adjustcorneramount], poly))
        else:
            notmod = False

    area = 0
    #Adjust based on startAt
    if startAt == 1 and len(arcpoints) > 0:
        arcpoint1 = arcpoints[0]
        arcpoints.pop(0)
        arcpoints.append(arcpoint1)

    #Sum arc areas
    for i in range(0, len(arcpoints), 2):
        area += getArcArea(arcpoints[i], arcpoints[i+1], center, abs(radius))
    area += getArea(intersectpoints)
    
    if len(arcpoints) == 0 and area == 0:
        if pointInPoly(center, poly):
            #circle lies entirely inside poly
            return radius*radius*math.pi, []
        #circle lies entirely outside poly
        return 0, []
    
    return area % (radius*radius*math.pi), arcpoints

#Newton's method to match a1, a2, a3s
#Matches area fractions a1, a2, a3
def getArcFacet(poly1, poly2, poly3, afrac1, afrac2, afrac3, epsilon, gcenterx=None, gcentery=None, gradius=None):
    #Basic sanity tests:
    assert afrac1 >= 0 and afrac1 <= 1 and afrac2 >= 0 and afrac2 <= 1 and afrac3 >= 0 and afrac3 <= 1, print("Given areas for circular facet are not valid")

    #Hyperparameters:
    scaleEpsilon = 0.99 #Adjust threshold by this amount per failed timestep
    dtbase = 1e-8 #Timestep used to calculate numerical estimates of derivatives
    fixLinearFacetOrientation = epsilon #Threshold used to determine orientation of wrong linear facets
    fixLinearFacet = 1.1 #Amount to adjust linear facet guess by to fix
    maxTimestep = 75 #Amount of timesteps allowed before we declare failure
    
    #Rotate so that x-axis is linear facet
    l1, l2 = getLinearFacet(poly1, poly3, afrac1, afrac3, epsilon/10)

    #Center to left of line l1 to l2
    if getPolyLineArea(poly2, l1, l2) > afrac2*getArea(poly2):
        retcenter, retradius, retintersects = getArcFacet(poly3, poly2, poly1, 1-afrac3, 1-afrac2, 1-afrac1, epsilon)
        if retcenter is not None:
            retintersects.reverse()
            retradius *= -1
            return retcenter, retradius, retintersects #TODO: radius should always be positive, negative radius means facet is concave, area fraction k -> 1-k
        else:
            return None, None, None

    #Convert area fractions to areas
    poly1area = getArea(poly1)
    poly2area = getArea(poly2)
    poly3area = getArea(poly3)
    a1 = afrac1*poly1area
    a2 = afrac2*poly2area
    a3 = afrac3*poly3area

    poly1intersects = getPolyLineIntersects(poly1, l1, l2)
    poly2intersects = getPolyLineIntersects(poly2, l1, l2)
    poly3intersects = getPolyLineIntersects(poly3, l1, l2)

    if len(poly2intersects) > 0 and (getDistance(poly2intersects[0], poly1intersects[-1]) > fixLinearFacetOrientation and getDistance(poly2intersects[-1], poly3intersects[0]) > fixLinearFacetOrientation):
        #Not a good linear facet
        print("Not a good linear facet")
        if getDistance(poly2intersects[-1], poly1intersects[0]) < fixLinearFacetOrientation:
            #Linear facet intersects poly2, poly1, poly3
            invertpoint = poly2intersects[-1]
            for p in poly2:
                if p in poly1:
                    invertpoint2 = p
                    break
            nl0 = l1[0] + fixLinearFacet*((invertpoint[0]-l1[0]) - ((invertpoint[0]-l1[0])*(invertpoint2[0]-invertpoint[0]) + (invertpoint[1]-l1[1])*(invertpoint2[1]-invertpoint[1]))/getDistance(invertpoint, invertpoint2)**2*(invertpoint2[0]-invertpoint[0]))
            nl1 = l1[1] + fixLinearFacet*((invertpoint[1]-l1[1]) - ((invertpoint[0]-l1[0])*(invertpoint2[0]-invertpoint[0]) + (invertpoint[1]-l1[1])*(invertpoint2[1]-invertpoint[1]))/getDistance(invertpoint, invertpoint2)**2*(invertpoint2[1]-invertpoint[1]))
            l1 = [nl0, nl1]
            nl0 = l2[0] + fixLinearFacet*((invertpoint[0]-l2[0]) - ((invertpoint[0]-l2[0])*(invertpoint2[0]-invertpoint[0]) + (invertpoint[1]-l2[1])*(invertpoint2[1]-invertpoint[1]))/getDistance(invertpoint, invertpoint2)**2*(invertpoint2[0]-invertpoint[0]))
            nl1 = l2[1] + fixLinearFacet*((invertpoint[1]-l2[1]) - ((invertpoint[0]-l2[0])*(invertpoint2[0]-invertpoint[0]) + (invertpoint[1]-l2[1])*(invertpoint2[1]-invertpoint[1]))/getDistance(invertpoint, invertpoint2)**2*(invertpoint2[1]-invertpoint[1]))
            l2 = [nl0, nl1]
        elif getDistance(poly2intersects[0], poly3intersects[-1]) < fixLinearFacetOrientation:
            #Linear facet intersects poly1, poly3, poly2
            invertpoint = poly2intersects[0]
            for p in poly2:
                if p in poly3:
                    invertpoint2 = p
                    break
            nl0 = l1[0] + fixLinearFacet*((invertpoint[0]-l1[0]) - ((invertpoint[0]-l1[0])*(invertpoint2[0]-invertpoint[0]) + (invertpoint[1]-l1[1])*(invertpoint2[1]-invertpoint[1]))/getDistance(invertpoint, invertpoint2)**2*(invertpoint2[0]-invertpoint[0]))
            nl1 = l1[1] + fixLinearFacet*((invertpoint[1]-l1[1]) - ((invertpoint[0]-l1[0])*(invertpoint2[0]-invertpoint[0]) + (invertpoint[1]-l1[1])*(invertpoint2[1]-invertpoint[1]))/getDistance(invertpoint, invertpoint2)**2*(invertpoint2[1]-invertpoint[1]))
            l1 = [nl0, nl1]
            nl0 = l2[0] + fixLinearFacet*((invertpoint[0]-l2[0]) - ((invertpoint[0]-l2[0])*(invertpoint2[0]-invertpoint[0]) + (invertpoint[1]-l2[1])*(invertpoint2[1]-invertpoint[1]))/getDistance(invertpoint, invertpoint2)**2*(invertpoint2[0]-invertpoint[0]))
            nl1 = l2[1] + fixLinearFacet*((invertpoint[1]-l2[1]) - ((invertpoint[0]-l2[0])*(invertpoint2[0]-invertpoint[0]) + (invertpoint[1]-l2[1])*(invertpoint2[1]-invertpoint[1]))/getDistance(invertpoint, invertpoint2)**2*(invertpoint2[1]-invertpoint[1]))
            l2 = [nl0, nl1]
        else:
            #Unknown configuration
            print("Unknown arc facet configuration: getArcFacet({}, {}, {}, {}, {}, {}, {})".format(poly1, poly2, poly3, afrac1, afrac2, afrac3, epsilon))

    rot = lambda p : [((l2[0]-l1[0])*(p[0]-l1[0]) + (l2[1]-l1[1])*(p[1]-l1[1]))/getDistance(l1, l2), ((l1[1]-l2[1])*(p[0]-l1[0]) + (l2[0]-l1[0])*(p[1]-l1[1]))/getDistance(l1, l2)]
    unrot = lambda p : [((l2[0]-l1[0])*p[0] - (l2[1]-l1[1])*p[1])/getDistance(l1, l2) + l1[0], (-(l1[1]-l2[1])*p[0] + (l2[0]-l1[0])*p[1])/getDistance(l1, l2) + l1[1]]
    rpoly1 = list(map(rot, poly1))
    rpoly2 = list(map(rot, poly2))
    rpoly3 = list(map(rot, poly3))
    
    #Working in rotated frame
    #Find where x-axis intersects rpoly1, rpoly3
    intersects1 = []
    for i in range(len(rpoly1)):
        p1 = rpoly1[i]
        p2 = rpoly1[(i+1) % len(rpoly1)]
        if ((p1[1] > 0) != (p2[1] > 0)):
            intersects1.append(p1[0] + (p2[0]-p1[0])*p1[1]/(p1[1]-p2[1]))
    #towards r1down = cura1 decreases
    r1down = max(intersects1)
    r1up = min(intersects1)
    intersects3 = []
    for i in range(len(rpoly3)):
        p1 = rpoly3[i]
        p2 = rpoly3[(i+1) % len(rpoly3)]
        if ((p1[1] > 0) != (p2[1] > 0)):
            intersects3.append(p1[0] + (p2[0]-p1[0])*p1[1]/(p1[1]-p2[1]))
    #towards r3down = cura3 decreases
    r3down = min(intersects3)
    r3up = max(intersects3)
    

    #Idea: degrees of freedom are a, r1, r3
    #r1 lies between r1down and r1up, r3 lies between r3down and r3up
    #Adjust a to match a2, then adjust r1 to match a1, then adjust r3 to match a3, then repeat until convergence
    cura1 = float("inf")
    cura2 = float("inf")
    cura3 = float("inf")
    #higher t1, t3 = lower cura1, cura3 respectively

    t1 = 0.5
    t3 = 0.5
    converged = False
    r1 = r1down*t1 + r1up*(1-t1)
    r3 = r3down*t3 + r3up*(1-t3)
    radius = abs(r3-r1)
    center = [(r1+r3)/2, math.sqrt(radius**2 - ((r3-r1)**2) / 4)]
    
    numcycles = 0
    
    while (not(converged) and numcycles < maxTimestep):

        #adjust radius to match a2
        radius = abs(r3-r1)
        center = [(r1+r3)/2, math.sqrt(radius**2 - ((r3-r1)**2) / 4)]
        cura2, _ = getCircleIntersectArea(center, radius, rpoly2)
        rgap = radius
        doConverge = False
        #higher a = higher cura2
        largeenough = False
        innercycles = 0
        while abs(cura2 - a2)/poly2area > epsilon*(scaleEpsilon**numcycles) and innercycles < maxTimestep:
            innercycles += 1
            if not(largeenough):
                largeenough2 = True
                for rpoly2vertex in rpoly2:
                    if rpoly2vertex[1] > 0 and getDistance(rpoly2vertex, center) > radius:
                        largeenough2 = False
                if largeenough2:
                    largeenough = True
            if cura2 > a2:
                radius += rgap
                if not(doConverge):
                    rgap *= 2
                else:
                    rgap /= 2
            else:
                if not(largeenough):
                    radius += rgap
                    rgap *= 2
                elif not(doConverge):
                    rgap /= 4
                    radius -= rgap
                    doConverge = True
                    rgap /= 2
                else:
                    radius -= rgap
                    rgap /= 2
            #Max range of radius hit, return error
            if radius**2 - ((r3-r1)**2) / 4 < 0:
                print("Error in getArcFacet({}, {}, {}, {}, {}, {}, {})".format(poly1, poly2, poly3, afrac1, afrac2, afrac3, epsilon))
                return None, None, None
            center = [(r1+r3)/2, math.sqrt(radius**2 - ((r3-r1)**2) / 4)]
            cura2, _ = getCircleIntersectArea(center, radius, rpoly2)

        #adjust r1 to match a1
        dt = dtbase
        center = [(r1+r3)/2, math.sqrt(radius**2 - ((r3-r1)**2) / 4)]
        cura1, _ = getCircleIntersectArea(center, radius, rpoly1)

        innercycles = 0
        while abs(cura1 - a1)/poly1area > epsilon*(scaleEpsilon**numcycles) and innercycles < maxTimestep:
            innercycles += 1
            #Max range of a1 hit, break and continue
            if radius**2 - ((r3-r1-dt)**2) / 4 < 0:
                break
            cura1plusdt, _ = getCircleIntersectArea([(r1+dt+r3)/2, math.sqrt(radius**2 - ((r3-r1-dt)**2) / 4)], radius, rpoly1)
            
            da1dr1 = (cura1plusdt-cura1)/dt
            #Numerical derivative is 0, return error
            if da1dr1 == 0:
                print("Error in getArcFacet({}, {}, {}, {}, {}, {}, {})".format(poly1, poly2, poly3, afrac1, afrac2, afrac3, epsilon))
                return None, None, None
            r1 = max(r3 - 2*radius, r1 + (a1-cura1)/da1dr1)
            center = [(r1+r3)/2, math.sqrt(radius**2 - ((r3-r1)**2) / 4)]
            cura1, _ = getCircleIntersectArea(center, radius, rpoly1)
        
        #adjust radius to match a2
        radius = abs(r3-r1)
        center = [(r1+r3)/2, math.sqrt(radius**2 - ((r3-r1)**2) / 4)]
        cura2, _ = getCircleIntersectArea(center, radius, rpoly2)
        rgap = radius
        doConverge = False
        #higher a = higher cura2
        largeenough = False
        innercycles = 0
        while abs(cura2 - a2)/poly2area > epsilon*(scaleEpsilon**numcycles) and innercycles < maxTimestep:
            innercycles += 1
            if not(largeenough):
                largeenough2 = True
                for rpoly2vertex in rpoly2:
                    if rpoly2vertex[1] > 0 and getDistance(rpoly2vertex, center) > radius:
                        largeenough2 = False
                if largeenough2:
                    largeenough = True
            if cura2 > a2:
                radius += rgap
                if not(doConverge):
                    rgap *= 2
                else:
                    rgap /= 2
            else:
                if not(largeenough):
                    radius += rgap
                    rgap *= 2
                elif not(doConverge):
                    rgap /= 4
                    radius -= rgap
                    doConverge = True
                    rgap /= 2
                else:
                    radius -= rgap
                    rgap /= 2
            #Max range of radius hit, return error
            if radius**2 - ((r3-r1)**2) / 4 < 0:
                print("Error in getArcFacet({}, {}, {}, {}, {}, {}, {})".format(poly1, poly2, poly3, afrac1, afrac2, afrac3, epsilon))
                return None, None, None
            center = [(r1+r3)/2, math.sqrt(radius**2 - ((r3-r1)**2) / 4)]
            cura2, _ = getCircleIntersectArea(center, radius, rpoly2)

        #adjust r3 to match a3
        dt = dtbase
        center = [(r1+r3)/2, math.sqrt(radius**2 - ((r3-r1)**2) / 4)]
        cura3, _ = getCircleIntersectArea(center, radius, rpoly3)
        
        innercycles = 0
        while abs(cura3 - a3)/poly3area > epsilon*(scaleEpsilon**numcycles) and innercycles < maxTimestep:
            innercycles += 1
            #Max range of a3 hit, break and continue
            if radius**2 - ((r3-dt-r1)**2) / 4 < 0:
                break
            cura3minusdt, _ = getCircleIntersectArea([(r1+r3-dt)/2, math.sqrt(radius**2 - ((r3-dt-r1)**2) / 4)], radius, rpoly3)
            
            da3dr3 = (cura3-cura3minusdt)/dt
            #Numerical derivative is 0, return error
            if da3dr3 == 0:
                print("Error in getArcFacet({}, {}, {}, {}, {}, {}, {})".format(poly1, poly2, poly3, afrac1, afrac2, afrac3, epsilon))
                return None, None, None
            r3 = min(r1 + 2*radius, r3 + (a3-cura3)/da3dr3)
            center = [(r1+r3)/2, math.sqrt(radius**2 - ((r3-r1)**2) / 4)]
            cura3, _ = getCircleIntersectArea(center, radius, rpoly3)
        
        numcycles += 1
        
        #check for convergence
        cura1, _ = getCircleIntersectArea(center, radius, rpoly1)
        cura2, facetintersects = getCircleIntersectArea(center, radius, rpoly2)
        cura3, _ = getCircleIntersectArea(center, radius, rpoly3)
        if abs(cura1 - a1)/poly1area < epsilon and abs(cura2 - a2)/poly2area < epsilon and abs(cura3 - a3)/poly3area < epsilon:
            #print(numcycles)
            returnintersects = [unrot(facetintersect) for _,facetintersect in sorted(zip(list(map(lambda point: point[0], facetintersects)), facetintersects))]
            return unrot(center), radius, returnintersects

    #Max timesteps reached: return failure
    print("Max timesteps reached in getArcFacet({}, {}, {}, {}, {}, {}, {})".format(poly1, poly2, poly3, afrac1, afrac2, afrac3, epsilon))
    return None, None, None

#Assume edges are shared between poly1 and poly2, poly2 and poly3
def getArcFacetNewton2(poly1, poly2, poly3, a1, a2, a3, epsilon):

    #Rotate so that x-axis is linear facet
    l1, l2 = getLinearFacet(poly1, poly3, a1, a3, epsilon/10)

    #Center to left of line l1 to l2
    if getPolyLineArea(poly2, l1, l2) > a2*getArea(poly2):
        retcenter, retradius, retintersects = getArcFacetNewton2(poly3, poly2, poly1, 1-a3, 1-a2, 1-a1, epsilon)
        if retcenter is not None:
            retintersects.reverse()
            retradius *= -1
            return retcenter, retradius, retintersects #TODO: radius should always be positive, negative radius means facet is concave, area fraction k -> 1-k
        else:
            return None, None, None

    #Find edges
    for i in range(len(poly2)):
        for j in range(len(poly1)):
            if poly2[(i+1) % len(poly2)] == poly1[j] and poly2[i] == poly1[(j+1) % len(poly1)]:
                t1up = poly2[i]
                t1down = poly1[j]
        for j in range(len(poly3)):
            if poly2[(i+1) % len(poly2)] == poly3[j] and poly2[i] == poly3[(j+1) % len(poly1)]:
                t3up = poly3[j]
                t3down = poly2[i]

    #Use normal for linear facet
    linear_intersect1, linear_intersect2 = getLinearFacetFromNormal(poly1, a1, [-(l2[1]-l1[1]), l2[0]-l1[0]], epsilon/10)
    midpoint1 = lerp(linear_intersect1, linear_intersect2, 0.5)
    linear_intersect1, linear_intersect2 = getLinearFacetFromNormal(poly2, a2, [-(l2[1]-l1[1]), l2[0]-l1[0]], epsilon/10)
    midpoint2 = lerp(linear_intersect1, linear_intersect2, 0.5)
    linear_intersect1, linear_intersect2 = getLinearFacetFromNormal(poly3, a3, [-(l2[1]-l1[1]), l2[0]-l1[0]], epsilon/10)
    midpoint3 = lerp(linear_intersect1, linear_intersect2, 0.5)
    #print(midpoint1)
    #print(midpoint2)
    #print(midpoint3)

    #Get circumcircle
    circumcenter, circumradius = getCircumcircle(midpoint1, midpoint2, midpoint3)
    t1guesses = getCircleLineIntersects(t1up, t1down, circumcenter, circumradius)
    #print(t1guesses)
    r1 = getDistance(t1guesses[0], t1up)/getDistance(t1up, t1down)
    t3guesses = getCircleLineIntersects(t3up, t3down, circumcenter, circumradius)
    #print(t3guesses)
    r3 = getDistance(t3guesses[0], t3up)/getDistance(t3up, t3down)
    
    #Variables: r1, r3, radius

    #Hyperparameters:
    scaleEpsilon = 1 #Adjust threshold by this amount per failed timestep
    dt = 1e-8 #Timestep used to calculate numerical estimates of derivatives
    newtonFactor = 3e-1 #Move this amount in Newton direction
    maxTimestep = 5000 #Amount of timesteps allowed before we declare failure

    doPrint = False
    newtonType = 1

    #Convert area fractions to areas
    poly1area = getArea(poly1)
    poly2area = getArea(poly2)
    poly3area = getArea(poly3)
    a1 *= poly1area
    a2 *= poly2area
    a3 *= poly3area
    
    #Initial guess    
    converged = False
    numcycles = 0

    #Compute areas
    t1 = lerp(t1up, t1down, r1)
    t3 = lerp(t3up, t3down, r3)

    #if numcycles == 0 or abs(cura2 - a2)/(poly2area) > max(abs(cura1 - a1)/poly1area, abs(cura3 - a3)/poly3area):
    poly2_arc_area = a2 - getPolyLineArea(poly2, t1, t3)
    radius = matchArcArea(getDistance(t1, t3), poly2_arc_area, epsilon/10)

    if radius < 0 or radius == float('inf'):
        radius = 1e6

    center = getCenter(t1, t3, radius)

    cura1, _ = getCircleIntersectArea(center, radius, poly1)
    cura2, facetintersects = getCircleIntersectArea(center, radius, poly2)
    cura3, _ = getCircleIntersectArea(center, radius, poly3)
    
    cura1err = abs(cura1 - a1)/poly1area
    cura2err = abs(cura2 - a2)/poly2area
    cura3err = abs(cura3 - a3)/poly3area

    if doPrint:
        print("{}, {}, {}".format(r1, r3, radius))
        print("{}, {}, {}".format(cura1err, cura2err, cura3err))

    if cura1err < epsilon and cura2err < epsilon and cura3err < epsilon:
        converged = True

    while (not(converged) and numcycles < maxTimestep):

        #Compute derivatives (variables: t1, t3)
        if newtonType == 1:
            """
            t1plusdt = lerp(t1up, t1down, r1+dt/2)
            rt1plusdt = matchArcArea(getDistance(t1plusdt, t3), poly2_arc_area, epsilon/10)
            xplusdt = getCenter(t1plusdt, t3, rt1plusdt)
            cura1plusdt, _ = getCircleIntersectArea(xplusdt, rt1plusdt, poly1)
            cura3plusdt, _ = getCircleIntersectArea(xplusdt, rt1plusdt, poly3)
            t1minusdt = lerp(t1up, t1down, r1-dt/2)
            rt1minusdt = matchArcArea(getDistance(t1minusdt, t3), poly2_arc_area, epsilon/10)
            xminusdt = getCenter(t1minusdt, t3, rt1minusdt)
            cura1minusdt, _ = getCircleIntersectArea(xminusdt, rt1minusdt, poly1)
            cura3minusdt, _ = getCircleIntersectArea(xminusdt, rt1minusdt, poly3)
            
            da1dx = (cura1plusdt-cura1minusdt)/dt
            da3dx = (cura3plusdt-cura3minusdt)/dt
            
            t3plusdt = lerp(t3up, t3down, r3+dt/2)
            rt3plusdt = matchArcArea(getDistance(t1, t3plusdt), poly2_arc_area, epsilon/10)
            yplusdt = getCenter(t1, t3plusdt, rt3plusdt)
            cura1plusdt, _ = getCircleIntersectArea(yplusdt, rt3plusdt, poly1)
            cura3plusdt, _ = getCircleIntersectArea(yplusdt, rt3plusdt, poly3)

            t3minusdt = lerp(t3up, t3down, r3-dt/2)
            rt3minusdt = matchArcArea(getDistance(t1, t3minusdt), poly2_arc_area, epsilon/10)
            yminusdt = getCenter(t1, t3minusdt, rt3minusdt)
            cura1minusdt, _ = getCircleIntersectArea(yminusdt, rt3minusdt, poly1)
            cura3minusdt, _ = getCircleIntersectArea(yminusdt, rt3minusdt, poly3)

            da1dy = (cura1plusdt-cura1minusdt)/dt
            da3dy = (cura3plusdt-cura3minusdt)/dt
            """

            t1plusdt = lerp(t1up, t1down, r1+dt/2)
            xplusdt = getCenter(t1plusdt, t3, radius)
            cura1plusdt, _ = getCircleIntersectArea(xplusdt, radius, poly1)
            cura3plusdt, _ = getCircleIntersectArea(xplusdt, radius, poly3)
            t1minusdt = lerp(t1up, t1down, r1-dt/2)
            xminusdt = getCenter(t1minusdt, t3, radius)
            cura1minusdt, _ = getCircleIntersectArea(xminusdt, radius, poly1)
            cura3minusdt, _ = getCircleIntersectArea(xminusdt, radius, poly3)
            
            da1dx = (cura1plusdt-cura1minusdt)/dt
            da3dx = (cura3plusdt-cura3minusdt)/dt

            t3plusdt = lerp(t3up, t3down, r3+dt/2)
            yplusdt = getCenter(t1, t3plusdt, radius)
            cura1plusdt, _ = getCircleIntersectArea(yplusdt, radius, poly1)
            cura3plusdt, _ = getCircleIntersectArea(yplusdt, radius, poly3)
            t3minusdt = lerp(t3up, t3down, r3-dt/2)
            yminusdt = getCenter(t1, t3minusdt, radius)
            cura1minusdt, _ = getCircleIntersectArea(yminusdt, radius, poly1)
            cura3minusdt, _ = getCircleIntersectArea(yminusdt, radius, poly3)

            da1dy = (cura1plusdt-cura1minusdt)/dt
            da3dy = (cura3plusdt-cura3minusdt)/dt

        else:
            t1plusdt = lerp(t1up, t1down, r1+dt/2)
            t1minusdt = lerp(t1up, t1down, r1-dt/2)
            t3plusdt = lerp(t3up, t3down, r3+dt/2)
            t3minusdt = lerp(t3up, t3down, r3-dt/2)

            rt1plusdt = matchArcArea(getDistance(t1plusdt, t3plusdt), poly2_arc_area, epsilon/10)
            xplusdt = getCenter(t1plusdt, t3plusdt, rt1plusdt)
            cura1plusdt, _ = getCircleIntersectArea(xplusdt, rt1plusdt, poly1)
            cura3plusdt, _ = getCircleIntersectArea(xplusdt, rt1plusdt, poly3)
            rt1minusdt = matchArcArea(getDistance(t1minusdt, t3minusdt), poly2_arc_area, epsilon/10)
            xminusdt = getCenter(t1minusdt, t3minusdt, rt1minusdt)
            cura1minusdt, _ = getCircleIntersectArea(xminusdt, rt1minusdt, poly1)
            cura3minusdt, _ = getCircleIntersectArea(xminusdt, rt1minusdt, poly3)
            
            da1dx = (cura1plusdt-cura1minusdt)/(2*dt)
            da3dx = (cura3plusdt-cura3minusdt)/(2*dt)
            
            rt3plusdt = matchArcArea(getDistance(t1plusdt, t3minusdt), poly2_arc_area, epsilon/10)
            yplusdt = getCenter(t1plusdt, t3minusdt, rt3plusdt)
            cura1plusdt, _ = getCircleIntersectArea(yplusdt, rt3plusdt, poly1)
            cura3plusdt, _ = getCircleIntersectArea(yplusdt, rt3plusdt, poly3)

            rt3minusdt = matchArcArea(getDistance(t1minusdt, t3plusdt), poly2_arc_area, epsilon/10)
            yminusdt = getCenter(t1minusdt, t3plusdt, rt3minusdt)
            cura1minusdt, _ = getCircleIntersectArea(yminusdt, rt3minusdt, poly1)
            cura3minusdt, _ = getCircleIntersectArea(yminusdt, rt3minusdt, poly3)

            da1dy = (cura1plusdt-cura1minusdt)/(2*dt)
            da3dy = (cura3plusdt-cura3minusdt)/(2*dt)
        
        #print("{}, {}, {}, {}".format(da1dx, da3dx, da1dy, da3dy))

        #Newton's
        if newtonType == 1: #max(abs(cura1 - a1)/poly1area, abs(cura2 - a2)/poly2area, abs(cura3 - a3)/poly3area) < 1e-2:
            #print(numcycles)
            jacobian = np.array([[da1dx, da1dy], [da3dx, da3dy]])
            det = np.linalg.det(jacobian)
            assert det != 0
            jacobianinv = np.linalg.inv(jacobian)
            r1update = jacobianinv[0][0]*(a1-cura1) + jacobianinv[0][1]*(a3-cura3)
            r3update = jacobianinv[1][0]*(a1-cura1) + jacobianinv[1][1]*(a3-cura3)

        else:
            jacobian = np.array([[da1dx, da1dy], [da3dx, da3dy]])
            det = np.linalg.det(jacobian)
            assert det != 0
            jacobianinv = np.linalg.inv(jacobian)
            r1update = (jacobianinv[0][0]*(a1-cura1) + jacobianinv[0][1]*(a3-cura3) + jacobianinv[1][0]*(a1-cura1) + jacobianinv[1][1]*(a3-cura3))/2
            r3update = (jacobianinv[0][0]*(a1-cura1) + jacobianinv[0][1]*(a3-cura3) - jacobianinv[1][0]*(a1-cura1) - jacobianinv[1][1]*(a3-cura3))/2

        #Gradient descent
        """
        else:
            dr1 = da1dx*(a1-cura1) + da3dx*(a3-cura3)
            dr3 = da1dy*(a1-cura1) + da3dy*(a3-cura3)
            if dr1 > 0:
                r1k = (1-r1)/(2*dr1)
            else:
                r1k = r1/(2*abs(dr1))
            if dr3 > 0:
                r3k = (1-r3)/(2*dr3)
            else:
                r3k = r3/(2*abs(dr3))
            #gradient_constant = min(1e-3, 1/max(abs(da1dx), abs(da3dx), abs(da1dy), abs(da3dy)))
            gradient_constant = min(1e-3, r1k, r3k)
            r1 += gradient_constant*dr1
            r3 += gradient_constant*dr3
        """

        #radius = matchArcArea(getDistance(t1, t3), poly2_arc_area, epsilon/10)
        numcycles += 1

        #Compute proper Newton factor
        for newton_order in range(8):
            localNewton = newtonFactor * 10**(-newton_order)

            localr1 = r1 + localNewton*r1update
            localr3 = r3 + localNewton*r3update

            #Compute areas
            t1 = lerp(t1up, t1down, localr1)
            t3 = lerp(t3up, t3down, localr3)

            poly2_arc_area = a2 - getPolyLineArea(poly2, t1, t3)
            radius = matchArcArea(getDistance(t1, t3), poly2_arc_area, epsilon/10)

            if radius < 0 or radius == float('inf'):
                radius = 1e6

            center = getCenter(t1, t3, radius)

            cura1, _ = getCircleIntersectArea(center, radius, poly1)
            cura2, facetintersects = getCircleIntersectArea(center, radius, poly2)
            cura3, _ = getCircleIntersectArea(center, radius, poly3)

            localcura1err = abs(cura1 - a1)/poly1area
            localcura2err = abs(cura2 - a2)/poly2area
            localcura3err = abs(cura3 - a3)/poly3area
            
            if localcura1err < cura1err or localcura3err < cura3err:
                break
        
        if doPrint:
            print("{}, {}, {}".format(r1, r3, radius))
            print("{}, {}, {}".format(localcura1err, localcura2err, localcura3err))

        if localcura1err < epsilon and localcura2err < epsilon and localcura3err < epsilon:
            converged = True
        else:
            r1 = localr1
            r3 = localr3
            cura1err = localcura1err
            cura2err = localcura2err
            cura3err = localcura3err
    
    if converged:
        print(numcycles)
        if numcycles > 100:
            print("getArcFacetNewton2({}, {}, {}, {}, {}, {}, {})".format(poly1, poly2, poly3, a1/getArea(poly1), a2/getArea(poly2), a3/getArea(poly3), epsilon))
        poly1centroid = getCentroid(poly1)
        returnintersects = [facetintersect for _,facetintersect in sorted(zip(list(map(lambda point: getDistance(poly1centroid, point), facetintersects)), facetintersects))]
        return center, radius, returnintersects
    else:
        print("Max timesteps reached in getArcFacetNewton2({}, {}, {}, {}, {}, {}, {})".format(poly1, poly2, poly3, a1, a2, a3, epsilon))
        return None, None, None

#Assume edges are shared between poly1 and poly2, poly2 and poly3
def getArcFacetNewtonSigmoid(poly1, poly2, poly3, a1, a2, a3, epsilon):

    #Rotate so that x-axis is linear facet
    l1, l2 = getLinearFacet(poly1, poly3, a1, a3, epsilon/10)

    #Center to left of line l1 to l2
    if getPolyLineArea(poly2, l1, l2) > a2*getArea(poly2):
        retcenter, retradius, retintersects = getArcFacetNewton2(poly3, poly2, poly1, 1-a3, 1-a2, 1-a1, epsilon)
        if retcenter is not None:
            retintersects.reverse()
            retradius *= -1
            return retcenter, retradius, retintersects #TODO: radius should always be positive, negative radius means facet is concave, area fraction k -> 1-k
        else:
            return None, None, None

    #Find edges
    for i in range(len(poly2)):
        for j in range(len(poly1)):
            if poly2[(i+1) % len(poly2)] == poly1[j] and poly2[i] == poly1[(j+1) % len(poly1)]:
                t1up = poly2[i]
                t1down = poly1[j]
        for j in range(len(poly3)):
            if poly2[(i+1) % len(poly2)] == poly3[j] and poly2[i] == poly3[(j+1) % len(poly1)]:
                t3up = poly3[j]
                t3down = poly2[i]

    #Use normal for linear facet
    linear_intersect1, linear_intersect2 = getLinearFacetFromNormal(poly1, a1, [-(l2[1]-l1[1]), l2[0]-l1[0]], epsilon/10)
    midpoint1 = lerp(linear_intersect1, linear_intersect2, 0.5)
    linear_intersect1, linear_intersect2 = getLinearFacetFromNormal(poly2, a2, [-(l2[1]-l1[1]), l2[0]-l1[0]], epsilon/10)
    midpoint2 = lerp(linear_intersect1, linear_intersect2, 0.5)
    linear_intersect1, linear_intersect2 = getLinearFacetFromNormal(poly3, a3, [-(l2[1]-l1[1]), l2[0]-l1[0]], epsilon/10)
    midpoint3 = lerp(linear_intersect1, linear_intersect2, 0.5)
    #print(midpoint1)
    #print(midpoint2)
    #print(midpoint3)

    #Get circumcircle
    circumcenter, circumradius = getCircumcircle(midpoint1, midpoint2, midpoint3)
    t1guesses = getCircleLineIntersects(t1up, t1down, circumcenter, circumradius)
    #print(t1guesses)
    r1 = getDistance(t1guesses[0], t1up)/getDistance(t1up, t1down)
    t3guesses = getCircleLineIntersects(t3up, t3down, circumcenter, circumradius)
    #print(t3guesses)
    r3 = getDistance(t3guesses[0], t3up)/getDistance(t3up, t3down)
    
    #Variables: r1, r3, radius

    x1 = -math.log(1/r1 - 1)
    x3 = -math.log(1/r3 - 1)

    #Hyperparameters:
    scaleEpsilon = 1 #Adjust threshold by this amount per failed timestep
    dt = 1e-8 #Timestep used to calculate numerical estimates of derivatives
    newtonFactor = 1 #Move this amount in Newton direction
    maxTimestep = 50 #Amount of timesteps allowed before we declare failure

    doPrint = True
    newtonType = 1

    #Convert area fractions to areas
    poly1area = getArea(poly1)
    poly2area = getArea(poly2)
    poly3area = getArea(poly3)
    a1 *= poly1area
    a2 *= poly2area
    a3 *= poly3area
    
    #Initial guess    
    converged = False
    numcycles = 0

    #Compute areas
    t1 = lerp(t1up, t1down, r1)
    t3 = lerp(t3up, t3down, r3)

    #if numcycles == 0 or abs(cura2 - a2)/(poly2area) > max(abs(cura1 - a1)/poly1area, abs(cura3 - a3)/poly3area):
    poly2_arc_area = a2 - getPolyLineArea(poly2, t1, t3)
    radius = matchArcArea(getDistance(t1, t3), poly2_arc_area, epsilon/10)

    if radius < 0 or radius == float('inf'):
        radius = 1e6

    center = getCenter(t1, t3, radius)

    cura1, _ = getCircleIntersectArea(center, radius, poly1)
    cura2, facetintersects = getCircleIntersectArea(center, radius, poly2)
    cura3, _ = getCircleIntersectArea(center, radius, poly3)
    
    cura1err = abs(cura1 - a1)/poly1area
    cura2err = abs(cura2 - a2)/poly2area
    cura3err = abs(cura3 - a3)/poly3area

    if doPrint:
        print("{}, {}, {}".format(r1, r3, radius))
        print("{}, {}, {}".format(cura1err, cura2err, cura3err))

    if cura1err < epsilon and cura2err < epsilon and cura3err < epsilon:
        converged = True

    sigmoid = lambda x : 1/(1+math.exp(-x))

    while (not(converged) and numcycles < maxTimestep):

        #Compute derivatives (variables: t1, t3)
        if newtonType == 1:
            """
            t1plusdt = lerp(t1up, t1down, r1+dt/2)
            rt1plusdt = matchArcArea(getDistance(t1plusdt, t3), poly2_arc_area, epsilon/10)
            xplusdt = getCenter(t1plusdt, t3, rt1plusdt)
            cura1plusdt, _ = getCircleIntersectArea(xplusdt, rt1plusdt, poly1)
            cura3plusdt, _ = getCircleIntersectArea(xplusdt, rt1plusdt, poly3)
            t1minusdt = lerp(t1up, t1down, r1-dt/2)
            rt1minusdt = matchArcArea(getDistance(t1minusdt, t3), poly2_arc_area, epsilon/10)
            xminusdt = getCenter(t1minusdt, t3, rt1minusdt)
            cura1minusdt, _ = getCircleIntersectArea(xminusdt, rt1minusdt, poly1)
            cura3minusdt, _ = getCircleIntersectArea(xminusdt, rt1minusdt, poly3)
            
            da1dx = (cura1plusdt-cura1minusdt)/dt
            da3dx = (cura3plusdt-cura3minusdt)/dt
            
            t3plusdt = lerp(t3up, t3down, r3+dt/2)
            rt3plusdt = matchArcArea(getDistance(t1, t3plusdt), poly2_arc_area, epsilon/10)
            yplusdt = getCenter(t1, t3plusdt, rt3plusdt)
            cura1plusdt, _ = getCircleIntersectArea(yplusdt, rt3plusdt, poly1)
            cura3plusdt, _ = getCircleIntersectArea(yplusdt, rt3plusdt, poly3)

            t3minusdt = lerp(t3up, t3down, r3-dt/2)
            rt3minusdt = matchArcArea(getDistance(t1, t3minusdt), poly2_arc_area, epsilon/10)
            yminusdt = getCenter(t1, t3minusdt, rt3minusdt)
            cura1minusdt, _ = getCircleIntersectArea(yminusdt, rt3minusdt, poly1)
            cura3minusdt, _ = getCircleIntersectArea(yminusdt, rt3minusdt, poly3)

            da1dy = (cura1plusdt-cura1minusdt)/dt
            da3dy = (cura3plusdt-cura3minusdt)/dt
            """

            t1plusdt = lerp(t1up, t1down, sigmoid(x1+dt/2))
            t1minusdt = lerp(t1up, t1down, sigmoid(x1-dt/2))
            
            #t1plusdt = lerp(t1up, t1down, r1+dt/2)
            xplusdt = getCenter(t1plusdt, t3, radius)
            cura1plusdt, _ = getCircleIntersectArea(xplusdt, radius, poly1)
            cura2plusdt, _ = getCircleIntersectArea(xplusdt, radius, poly2)
            cura3plusdt, _ = getCircleIntersectArea(xplusdt, radius, poly3)
            #t1minusdt = lerp(t1up, t1down, r1-dt/2)
            xminusdt = getCenter(t1minusdt, t3, radius)
            cura1minusdt, _ = getCircleIntersectArea(xminusdt, radius, poly1)
            cura3minusdt, _ = getCircleIntersectArea(xminusdt, radius, poly3)
            
            da1dx = (cura1plusdt-cura1minusdt)/dt
            da3dx = (cura3plusdt-cura3minusdt)/dt
            
            t3plusdt = lerp(t3up, t3down, sigmoid(x3+dt/2))
            t3minusdt = lerp(t3up, t3down, sigmoid(x3-dt/2))

            #t3plusdt = lerp(t3up, t3down, r3+dt/2)
            yplusdt = getCenter(t1, t3plusdt, radius)
            cura1plusdt, _ = getCircleIntersectArea(yplusdt, radius, poly1)
            cura2plusdt, _ = getCircleIntersectArea(yplusdt, radius, poly2)
            cura3plusdt, _ = getCircleIntersectArea(yplusdt, radius, poly3)
            #t3minusdt = lerp(t3up, t3down, r3-dt/2)
            yminusdt = getCenter(t1, t3minusdt, radius)
            cura1minusdt, _ = getCircleIntersectArea(yminusdt, radius, poly1)
            cura3minusdt, _ = getCircleIntersectArea(yminusdt, radius, poly3)

            da1dy = (cura1plusdt-cura1minusdt)/dt
            da3dy = (cura3plusdt-cura3minusdt)/dt
            
        else:
            t1plusdt = lerp(t1up, t1down, r1+dt/2)
            t1minusdt = lerp(t1up, t1down, r1-dt/2)
            t3plusdt = lerp(t3up, t3down, r3+dt/2)
            t3minusdt = lerp(t3up, t3down, r3-dt/2)

            rt1plusdt = matchArcArea(getDistance(t1plusdt, t3plusdt), poly2_arc_area, epsilon/10)
            xplusdt = getCenter(t1plusdt, t3plusdt, rt1plusdt)
            cura1plusdt, _ = getCircleIntersectArea(xplusdt, rt1plusdt, poly1)
            cura3plusdt, _ = getCircleIntersectArea(xplusdt, rt1plusdt, poly3)
            rt1minusdt = matchArcArea(getDistance(t1minusdt, t3minusdt), poly2_arc_area, epsilon/10)
            xminusdt = getCenter(t1minusdt, t3minusdt, rt1minusdt)
            cura1minusdt, _ = getCircleIntersectArea(xminusdt, rt1minusdt, poly1)
            cura3minusdt, _ = getCircleIntersectArea(xminusdt, rt1minusdt, poly3)
            
            da1dx = (cura1plusdt-cura1minusdt)/(2*dt)
            da3dx = (cura3plusdt-cura3minusdt)/(2*dt)
            
            rt3plusdt = matchArcArea(getDistance(t1plusdt, t3minusdt), poly2_arc_area, epsilon/10)
            yplusdt = getCenter(t1plusdt, t3minusdt, rt3plusdt)
            cura1plusdt, _ = getCircleIntersectArea(yplusdt, rt3plusdt, poly1)
            cura3plusdt, _ = getCircleIntersectArea(yplusdt, rt3plusdt, poly3)

            rt3minusdt = matchArcArea(getDistance(t1minusdt, t3plusdt), poly2_arc_area, epsilon/10)
            yminusdt = getCenter(t1minusdt, t3plusdt, rt3minusdt)
            cura1minusdt, _ = getCircleIntersectArea(yminusdt, rt3minusdt, poly1)
            cura3minusdt, _ = getCircleIntersectArea(yminusdt, rt3minusdt, poly3)

            da1dy = (cura1plusdt-cura1minusdt)/(2*dt)
            da3dy = (cura3plusdt-cura3minusdt)/(2*dt)
        
        #print("{}, {}, {}, {}".format(da1dx, da3dx, da1dy, da3dy))

        #Newton's
        if newtonType == 1: #max(abs(cura1 - a1)/poly1area, abs(cura2 - a2)/poly2area, abs(cura3 - a3)/poly3area) < 1e-2:
            #print(numcycles)
            jacobian = np.array([[da1dx, da1dy], [da3dx, da3dy]])
            det = np.linalg.det(jacobian)
            assert det != 0
            jacobianinv = np.linalg.inv(jacobian)
            r1update = jacobianinv[0][0]*(a1-cura1) + jacobianinv[0][1]*(a3-cura3)
            r3update = jacobianinv[1][0]*(a1-cura1) + jacobianinv[1][1]*(a3-cura3)

        else:
            jacobian = np.array([[da1dx, da1dy], [da3dx, da3dy]])
            det = np.linalg.det(jacobian)
            assert det != 0
            jacobianinv = np.linalg.inv(jacobian)
            r1update = (jacobianinv[0][0]*(a1-cura1) + jacobianinv[0][1]*(a3-cura3) + jacobianinv[1][0]*(a1-cura1) + jacobianinv[1][1]*(a3-cura3))/2
            r3update = (jacobianinv[0][0]*(a1-cura1) + jacobianinv[0][1]*(a3-cura3) - jacobianinv[1][0]*(a1-cura1) - jacobianinv[1][1]*(a3-cura3))/2

        #Gradient descent
        """
        else:
            dr1 = da1dx*(a1-cura1) + da3dx*(a3-cura3)
            dr3 = da1dy*(a1-cura1) + da3dy*(a3-cura3)
            if dr1 > 0:
                r1k = (1-r1)/(2*dr1)
            else:
                r1k = r1/(2*abs(dr1))
            if dr3 > 0:
                r3k = (1-r3)/(2*dr3)
            else:
                r3k = r3/(2*abs(dr3))
            #gradient_constant = min(1e-3, 1/max(abs(da1dx), abs(da3dx), abs(da1dy), abs(da3dy)))
            gradient_constant = min(1e-3, r1k, r3k)
            r1 += gradient_constant*dr1
            r3 += gradient_constant*dr3
        """

        #radius = matchArcArea(getDistance(t1, t3), poly2_arc_area, epsilon/10)
        numcycles += 1

        #Compute proper Newton factor
        for newton_order in range(8):
            localNewton = newtonFactor * 10**(-newton_order)

            #localr1 = r1 + localNewton*r1update
            #localr3 = r3 + localNewton*r3update

            localx1 = x1+localNewton*r1update
            localr1 = sigmoid(localx1)
            localx3 = x3+localNewton*r3update
            localr3 = sigmoid(localx3)

            #Compute areas
            t1 = lerp(t1up, t1down, localr1)
            t3 = lerp(t3up, t3down, localr3)

            poly2_arc_area = a2 - getPolyLineArea(poly2, t1, t3)

            #radius = matchArcArea(getDistance(t1, t3), poly2_arc_area, epsilon/10)
            #if radius < 0 or radius == float('inf'):
            #    radius = 1e6

            center = getCenter(t1, t3, radius)

            cura1, _ = getCircleIntersectArea(center, radius, poly1)
            cura2, facetintersects = getCircleIntersectArea(center, radius, poly2)
            cura3, _ = getCircleIntersectArea(center, radius, poly3)

            localcura1err = abs(cura1 - a1)/poly1area
            localcura2err = abs(cura2 - a2)/poly2area
            localcura3err = abs(cura3 - a3)/poly3area
            
            if localcura1err < cura1err or localcura3err < cura3err:
                break

        print(localNewton)

        if doPrint:
            print("{}, {}, {}".format(r1, r3, radius))
            print("{}, {}, {}".format((cura1 - a1)/poly1area, (cura2 - a2)/poly2area, (cura3 - a3)/poly3area))

        if localcura1err < epsilon and localcura2err < epsilon and localcura3err < epsilon:
            converged = True
        else:
            x1 = localx1
            x3 = localx3
            r1 = localr1
            r3 = localr3

            cura1err = localcura1err
            cura2err = localcura2err
            cura3err = localcura3err

        localr = radius
        num_radius_iters = 0
        while abs(cura1+cura2+cura3-a1-a2-a3) > epsilon/10 and num_radius_iters < 50:
            
            localrplusdt = localr+dt
            centerplusdt = getCenter(t1, t3, localrplusdt)
            cura1plusdt, _ = getCircleIntersectArea(centerplusdt, localrplusdt, poly1)
            cura2plusdt, facetintersects = getCircleIntersectArea(centerplusdt, localrplusdt, poly2)
            cura3plusdt, _ = getCircleIntersectArea(centerplusdt, localrplusdt, poly3)

            dlocalrdt = (cura1plusdt+cura2plusdt+cura3plusdt-cura1-cura2-cura3)/dt

            print(localr)
            localr += 1e-2*(a1+a2+a3-cura1-cura2-cura3)/dlocalrdt
            
            center = getCenter(t1, t3, localr)
            cura1, _ = getCircleIntersectArea(center, localr, poly1)
            cura2, facetintersects = getCircleIntersectArea(center, localr, poly2)
            cura3, _ = getCircleIntersectArea(center, localr, poly3)

            num_radius_iters += 1

        print(cura1+cura2+cura3-a1-a2-a3)

        matcha2r = matchArcArea(getDistance(t1, t3), poly2_arc_area, epsilon/10)
        if matcha2r < 0 or matcha2r == float('inf'):
            matcha2r = 1e6

        radius = 2/(1/matcha2r + 1/localr)
        center = getCenter(t1, t3, radius)
        cura1, _ = getCircleIntersectArea(center, radius, poly1)
        cura2, facetintersects = getCircleIntersectArea(center, radius, poly2)
        cura3, _ = getCircleIntersectArea(center, radius, poly3)
        

    
    if converged:
        print(numcycles)
        if numcycles > 1000:
            print("getArcFacetNewton2({}, {}, {}, {}, {}, {}, {})".format(poly1, poly2, poly3, a1/getArea(poly1), a2/getArea(poly2), a3/getArea(poly3), epsilon))
        poly1centroid = getCentroid(poly1)
        returnintersects = [facetintersect for _,facetintersect in sorted(zip(list(map(lambda point: getDistance(poly1centroid, point), facetintersects)), facetintersects))]
        return center, radius, returnintersects
    else:
        print("Max timesteps reached in getArcFacetNewton2({}, {}, {}, {}, {}, {}, {})".format(poly1, poly2, poly3, a1, a2, a3, epsilon))
        return None, None, None

#Newton's method, requires precise initial guess
#Matches area fractions a1, a2, a3
def getArcFacetNewton(poly1, poly2, poly3, a1, a2, a3, epsilon, gcenterx, gcentery, gradius):
    #Center to left of line l1 to l2
    if gradius < 0:
        retcenter, retradius, retintersects = getArcFacetNewton(poly3, poly2, poly1, 1-a3, 1-a2, 1-a1, epsilon, gcenterx, gcentery, -gradius)
        if retcenter is not None:
            retintersects.reverse()
            retradius *= -1
            return retcenter, retradius, retintersects #TODO: radius should always be positive, negative radius means facet is concave, area fraction k -> 1-k
        else:
            return None, None, None

    #Hyperparameters:
    scaleEpsilon = 1 #Adjust threshold by this amount per failed timestep
    dtbase = 1e-8 #Timestep used to calculate numerical estimates of derivatives
    newtonFactor = 1 #Move this amount in Newton direction
    maxTimestep = 500 #Amount of timesteps allowed before we declare failure
    
    #Convert area fractions to areas
    poly1area = getArea(poly1)
    poly2area = getArea(poly2)
    poly3area = getArea(poly3)
    a1 *= poly1area
    a2 *= poly2area
    a3 *= poly3area
    
    #Initial guess
    centerx = gcenterx
    centery = gcentery
    radius = gradius
    
    converged = False
    numcycles = 0

    while (not(converged) and numcycles < maxTimestep):
        #Compute areas
        center = [centerx, centery]
        cura1, _ = getCircleIntersectArea(center, radius, poly1)
        cura2, facetintersects = getCircleIntersectArea(center, radius, poly2)
        cura3, _ = getCircleIntersectArea(center, radius, poly3)
        if abs(cura1 - a1)/poly1area < epsilon and abs(cura2 - a2)/poly2area < epsilon and abs(cura3 - a3)/poly3area < epsilon:
            converged = True
        else:
            dt = dtbase
            
            #Compute derivatives
            xplusdt = [centerx+dt/2, centery]
            cura1plusdt, _ = getCircleIntersectArea(xplusdt, radius, poly1)
            cura2plusdt, _ = getCircleIntersectArea(xplusdt, radius, poly2)
            cura3plusdt, _ = getCircleIntersectArea(xplusdt, radius, poly3)
            xminusdt = [centerx-dt/2, centery]
            cura1minusdt, _ = getCircleIntersectArea(xminusdt, radius, poly1)
            cura2minusdt, _ = getCircleIntersectArea(xminusdt, radius, poly2)
            cura3minusdt, _ = getCircleIntersectArea(xminusdt, radius, poly3)
            
            da1dx = (cura1plusdt-cura1minusdt)/dt
            da2dx = (cura2plusdt-cura2minusdt)/dt
            da3dx = (cura3plusdt-cura3minusdt)/dt
            
            yplusdt = [centerx, centery+dt/2]
            cura1plusdt, _ = getCircleIntersectArea(yplusdt, radius, poly1)
            cura2plusdt, _ = getCircleIntersectArea(yplusdt, radius, poly2)
            cura3plusdt, _ = getCircleIntersectArea(yplusdt, radius, poly3)
            yminusdt = [centerx, centery-dt/2]
            cura1minusdt, _ = getCircleIntersectArea(yminusdt, radius, poly1)
            cura2minusdt, _ = getCircleIntersectArea(yminusdt, radius, poly2)
            cura3minusdt, _ = getCircleIntersectArea(yminusdt, radius, poly3)
            
            da1dy = (cura1plusdt-cura1minusdt)/dt
            da2dy = (cura2plusdt-cura2minusdt)/dt
            da3dy = (cura3plusdt-cura3minusdt)/dt
            
            cura1plusdt, _ = getCircleIntersectArea(center, radius+dt/2, poly1)
            cura2plusdt, _ = getCircleIntersectArea(center, radius+dt/2, poly2)
            cura3plusdt, _ = getCircleIntersectArea(center, radius+dt/2, poly3)
            cura1minusdt, _ = getCircleIntersectArea(center, radius-dt/2, poly1)
            cura2minusdt, _ = getCircleIntersectArea(center, radius-dt/2, poly2)
            cura3minusdt, _ = getCircleIntersectArea(center, radius-dt/2, poly3)
            
            da1dr = (cura1plusdt-cura1minusdt)/dt
            da2dr = (cura2plusdt-cura2minusdt)/dt
            da3dr = (cura3plusdt-cura3minusdt)/dt
            
            jacobian = np.array([[da1dx, da1dy, da1dr], [da2dx, da2dy, da2dr], [da3dx, da3dy, da3dr]])
            det = np.linalg.det(jacobian)
            
            assert det != 0
            
            jacobianinv = np.linalg.inv(jacobian)
            centerx += newtonFactor*(jacobianinv[0][0]*(a1-cura1) + jacobianinv[0][1]*(a2-cura2) + jacobianinv[0][2]*(a3-cura3))
            centery += newtonFactor*(jacobianinv[1][0]*(a1-cura1) + jacobianinv[1][1]*(a2-cura2) + jacobianinv[1][2]*(a3-cura3))
            radius += newtonFactor*(jacobianinv[2][0]*(a1-cura1) + jacobianinv[2][1]*(a2-cura2) + jacobianinv[2][2]*(a3-cura3))

        numcycles += 1
    
    if converged:
        poly1centroid = getCentroid(poly1)
        returnintersects = [facetintersect for _,facetintersect in sorted(zip(list(map(lambda point: getDistance(poly1centroid, point), facetintersects)), facetintersects))]
        return center, radius, returnintersects
    else:
        print("Max timesteps reached in getArcFacetNewton({}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(poly1, poly2, poly3, a1, a2, a3, epsilon, gcenterx, gcentery, gradius))
        return None, None, None

#Newton's method on cluster, requires precise initial guess
#polys = [poly1] + [cluster of polygons] + [poly3]
#Area fractions areas, shape corresponds to shape of polys
def getArcFacetNeighbors(polys, areas, epsilon, gcenterx, gcentery, gradius):
    #Center to left of line l1 to l2
    if gradius < 0:
        retcenter, retradius, retintersects = getArcFacetNewton(polys.reverse(), list(map(lambda x : 1-x, areas)).reverse(), epsilon, gcenterx, gcentery, -gradius)
        if retcenter is not None:
            retintersects.reverse()
            retradius *= -1
            return retcenter, retradius, retintersects #TODO: radius should always be positive, negative radius means facet is concave, area fraction k -> 1-k
        else:
            return None, None, None

    #Hyperparameters:
    scaleEpsilon = 1 #Adjust threshold by this amount per failed timestep
    dtbase = 1e-8 #Timestep used to calculate numerical estimates of derivatives
    newtonFactor = 1e-2 #Move this amount in Newton direction
    maxTimestep = 500 #Amount of timesteps allowed before we declare failure
    convthreshold = 1e-14 #If changes in area are all below this threshold, declare convergence (or failure)
    
    #Convert area fractions to areas
    curareas = list(map(lambda x : getCircleIntersectArea([gcenterx, gcentery], gradius, x)[0], polys)) #area fractions
    targetareas = list(map(lambda x : areas[x]*getArea(polys[x]), range(len(polys))))

    #Initial guess
    centerx = gcenterx
    centery = gcentery
    radius = gradius
    
    converged = False
    numcycles = 0

    while max(list(map(lambda x : abs(curareas[x] - targetareas[x])/getArea(polys[x]), range(len(polys))))) > epsilon and numcycles < maxTimestep:
        print(max(list(map(lambda x : abs(curareas[x] - targetareas[x])/getArea(polys[x]), range(len(polys))))))
        #Compute numerical derivatives
        curareasdx = list(map(lambda x : (getCircleIntersectArea([centerx+dtbase/2, centery], radius, x)[0]-getCircleIntersectArea([centerx-dtbase/2, centery], radius, x)[0])/dtbase/getArea(x), polys))
        curareasdy = list(map(lambda x : (getCircleIntersectArea([centerx, centery+dtbase/2], radius, x)[0]-getCircleIntersectArea([centerx, centery-dtbase/2], radius, x)[0])/dtbase/getArea(x), polys))
        curareasdr = list(map(lambda x : (getCircleIntersectArea([centerx, centery], radius+dtbase/2, x)[0]-getCircleIntersectArea([centerx, centery], radius-dtbase/2, x)[0])/dtbase/getArea(x), polys))
        #Jacobian: 3 by len(polys)
        jacobian = np.array([list(map(lambda x : curareasdx[x], range(len(polys)))), list(map(lambda x : curareasdy[x], range(len(polys)))), list(map(lambda x : curareasdr[x], range(len(polys))))])
        cornerchanges = np.matmul(np.matmul(np.linalg.inv(np.matmul(jacobian, np.transpose(jacobian))), jacobian), np.transpose(np.array(list(map(lambda x : targetareas[x]-curareas[x], range(len(polys)))))))
        centerx += newtonFactor*cornerchanges[0]
        centery += newtonFactor*cornerchanges[1]
        radius += newtonFactor*cornerchanges[2]
        newcurareas = list(map(lambda x : getCircleIntersectArea([centerx, centery], radius, x)[0], polys)) #area fractions
        #if max(list(map(lambda x : abs(curareas[x]-newcurareas[x]), range(len(polys))))) < convthreshold:
        #    break
        curareas = newcurareas
        numcycles += 1
        
    if max(list(map(lambda x : abs(curareas[x] - targetareas[x])/getArea(polys[x]), range(len(polys))))) < epsilon: #converged
        poly1centroid = getCentroid(polys[0])
        returnintersects = []
        for poly in polys[1:-1]:
            _, facetintersects = getCircleIntersectArea([centerx, centery], radius, poly)
            returnintersect = [facetintersect for _,facetintersect in sorted(zip(list(map(lambda point: getDistance(poly1centroid, point), facetintersects)), facetintersects))]
            returnintersects.append(returnintersect)
        return [centerx, centery], radius, returnintersects
    else:
        print("Max timesteps reached in getArcFacetNeighbors({}, {}, {}, {}, {}, {})".format(polys, areas, epsilon, gcenterx, gcentery, gradius))
        return None, None, None

#Assume edges are shared between poly1 and poly2, poly2 and poly3
def getArcFacetSigmoidRoot(poly1, poly2, poly3, a1, a2, a3, epsilon):

    #Rotate so that x-axis is linear facet
    l1, l2 = getLinearFacet(poly1, poly3, a1, a3, epsilon/10)

    #Center to left of line l1 to l2
    if getPolyLineArea(poly2, l1, l2) > a2*getArea(poly2):
        retcenter, retradius, retintersects = getArcFacetSigmoidRoot(poly3, poly2, poly1, 1-a3, 1-a2, 1-a1, epsilon)
        if retcenter is not None:
            retintersects.reverse()
            retradius *= -1
            return retcenter, retradius, retintersects #TODO: radius should always be positive, negative radius means facet is concave, area fraction k -> 1-k
        else:
            return None, None, None

    #Find edges
    for i in range(len(poly2)):
        for j in range(len(poly1)):
            if poly2[(i+1) % len(poly2)] == poly1[j] and poly2[i] == poly1[(j+1) % len(poly1)]:
                t1up = poly2[i]
                t1down = poly1[j]
        for j in range(len(poly3)):
            if poly2[(i+1) % len(poly2)] == poly3[j] and poly2[i] == poly3[(j+1) % len(poly1)]:
                t3up = poly3[j]
                t3down = poly2[i]

    #Use normal for linear facet
    linear_intersect1, linear_intersect2 = getLinearFacetFromNormal(poly1, a1, [-(l2[1]-l1[1]), l2[0]-l1[0]], epsilon/10)
    midpoint1 = lerp(linear_intersect1, linear_intersect2, 0.5)
    linear_intersect1, linear_intersect2 = getLinearFacetFromNormal(poly2, a2, [-(l2[1]-l1[1]), l2[0]-l1[0]], epsilon/10)
    midpoint2 = lerp(linear_intersect1, linear_intersect2, 0.5)
    linear_intersect1, linear_intersect2 = getLinearFacetFromNormal(poly3, a3, [-(l2[1]-l1[1]), l2[0]-l1[0]], epsilon/10)
    midpoint3 = lerp(linear_intersect1, linear_intersect2, 0.5)
    #print(midpoint1)
    #print(midpoint2)
    #print(midpoint3)

    #Get circumcircle
    circumcenter, circumradius = getCircumcircle(midpoint1, midpoint2, midpoint3)
    t1guesses = getCircleLineIntersects(t1up, t1down, circumcenter, circumradius)
    #print(t1guesses)
    r1 = getDistance(t1guesses[0], t1up)/getDistance(t1up, t1down)
    t3guesses = getCircleLineIntersects(t3up, t3down, circumcenter, circumradius)
    #print(t3guesses)
    r3 = getDistance(t3guesses[0], t3up)/getDistance(t3up, t3down)
    

    x1 = -math.log(1/r1 - 1)
    x3 = -math.log(1/r3 - 1)

    #Convert area fractions to areas
    poly1area = getArea(poly1)
    poly2area = getArea(poly2)
    poly3area = getArea(poly3)
    a1 *= poly1area
    a2 *= poly2area
    a3 *= poly3area

    sigmoid = lambda x : 1/(1+math.exp(-x))

    def rootFind(cur):
        x1 = cur[0]
        x3 = cur[1]
        t1 = lerp(t1up, t1down, sigmoid(x1))
        t3 = lerp(t3up, t3down, sigmoid(x3))
        poly2_arc_area = a2 - getPolyLineArea(poly2, t1, t3)
        radius = matchArcArea(getDistance(t1, t3), poly2_arc_area, epsilon/10)
        if radius < 0 or radius == float('inf'):
            radius = 1e6
        center = getCenter(t1, t3, radius)
        cura1, _ = getCircleIntersectArea(center, radius, poly1)
        cura3, _ = getCircleIntersectArea(center, radius, poly3)
        f = [(cura1-a1)/poly1area, (cura3-a3)/poly3area]
        return f
    
    guess = np.array([x1, x3])
    sol = root(rootFind, guess, method='hybr')
    retx1 = sol.x[0]
    retx3 = sol.x[1]
    t1 = lerp(t1up, t1down, sigmoid(retx1))
    t3 = lerp(t3up, t3down, sigmoid(retx3))
    poly2_arc_area = a2 - getPolyLineArea(poly2, t1, t3)
    radius = matchArcArea(getDistance(t1, t3), poly2_arc_area, epsilon/10)
    if radius < 0 or radius == float('inf'):
        radius = 1e6
    center = getCenter(t1, t3, radius)

    cura1, _ = getCircleIntersectArea(center, radius, poly1)
    cura2, facetintersects = getCircleIntersectArea(center, radius, poly2)
    cura3, _ = getCircleIntersectArea(center, radius, poly3)

    #print("{}, {}, {}".format((cura1-a1)/poly1area, (cura2-a2)/poly2area, (cura3-a3)/poly3area))

    centroid1 = getCentroid(poly1)

    poly1centroid = getCentroid(poly1)
    returnintersects = [facetintersect for _,facetintersect in sorted(zip(list(map(lambda point: getDistance(poly1centroid, point), facetintersects)), facetintersects))]
    return center, radius, returnintersects

def getArcFacetRoot(poly1, poly2, poly3, a1, a2, a3, epsilon, gcenterx=None, gcentery=None, gradius=None):
    #Basic sanity tests:
    assert a1 >= 0 and a1 <= 1 and a2 >= 0 and a2 <= 1 and a3 >= 0 and a3 <= 1, print("Given areas for circular facet are not valid")

    #If guess is none, form one:
    if gcenterx is None or gcentery is None or gradius is None:

        #Rotate so that x-axis is linear facet
        l1, l2 = getLinearFacet(poly1, poly3, a1, a3, epsilon/10)

        poly1intersects = getPolyLineIntersects(poly1, l1, l2)
        r1 = lerp(poly1intersects[0], poly1intersects[-1], 0.5)
        poly3intersects = getPolyLineIntersects(poly3, l1, l2)
        r3 = lerp(poly3intersects[0], poly3intersects[-1], 0.5)
        gradius = getDistance(r1, r3)
        if getPolyLineArea(poly2, l1, l2) > a2*getArea(poly2):
            gradius *= -1
        gcenter = getCenter(r1, r3, gradius)
            
        gcenterx = gcenter[0]
        gcentery = gcenter[1]

    truea1 = getArea(poly1)
    truea2 = getArea(poly2)
    truea3 = getArea(poly3)

    def rootFind(cur):
        curx = cur[0]
        cury = cur[1]
        curr = cur[2]
        center = [curx, cury]
        cura1, _ = getCircleIntersectArea(center, curr, poly1)
        cura2, _ = getCircleIntersectArea(center, curr, poly2)
        cura3, _ = getCircleIntersectArea(center, curr, poly3)
        f = [cura1/truea1 - a1, cura2/truea2 - a2, cura3/truea3 - a3]
        return f
    
    guess = np.array([gcenterx, gcentery, gradius])
    sol = root(rootFind, guess, method='hybr')
    center = [sol.x[0], sol.x[1]]
    radius = sol.x[2]
    _, intersects = getCircleIntersectArea(center, radius, poly2)
    centroid1 = getCentroid(poly1)

    #TODO: currently sorts poly2-circularfacet intersects by distance to centroid of poly1, problematic in some cases?
    try:
        l1, l3 = getLinearFacet(poly1, poly3, a1, a3, epsilon)
        l1 = lerp(l1, l3, -5)
        returnintersects = [facetintersect for _,facetintersect in sorted(zip(list(map(lambda point: getDistance(point, l1), intersects)), intersects))]
    except:
        returnintersects = [facetintersect for _,facetintersect in sorted(zip(list(map(lambda point: getDistance(point, centroid1), intersects)), intersects))]
    if len(intersects) > 0:
        return center, radius, returnintersects
    return None, None, None

def getArcFacetWithIntersects(poly, afrac, epsilon, arcintersect1, arcintersect2):
    #Basic sanity tests:
    assert afrac >= 0 and afrac <= 1, print("Given areas for circular facet are not valid")

    #Hyperparameters:
    scaleEpsilon = 0.99 #Adjust threshold by this amount per failed timestep
    dtbase = 1e-8 #Timestep used to calculate numerical estimates of derivatives
    maxTimestep = 75 #Amount of timesteps allowed before we declare failure

    polyarea = getArea(poly)
    area = afrac*polyarea

    if getPolyLineArea(poly, arcintersect1, arcintersect2) > area:
        try:
            center, radius, intersects = getArcFacetWithIntersects(poly, 1-afrac, epsilon, arcintersect2, arcintersect1)
            intersects.reverse()
            return center, -radius, intersects
        except:
            return None, None, None

    radius = getDistance(arcintersect1, arcintersect2)
    center = getCenter(arcintersect1, arcintersect2, radius)
    cura, _ = getCircleIntersectArea(center, radius, poly)
    rgap = radius
    doConverge = False
    #higher a = higher cura
    largeenough = False
    innercycles = 0
    while abs(cura - area)/polyarea > epsilon*(scaleEpsilon**innercycles) and innercycles < maxTimestep:
        innercycles += 1
        if not(largeenough):
            largeenough2 = True
            for vertex in poly:
                if not(pointRightOfLine(vertex, arcintersect1, arcintersect2)) and getDistance(vertex, center) > radius:
                    largeenough2 = False
            if largeenough2:
                largeenough = True
        if cura > area:
            radius += rgap
            if not(doConverge):
                rgap *= 2
            else:
                rgap /= 2
        else:
            if not(largeenough):
                radius += rgap
                rgap *= 2
            elif not(doConverge):
                rgap /= 4
                radius -= rgap
                doConverge = True
                rgap /= 2
            else:
                radius -= rgap
                rgap /= 2
        #Max range of radius hit, return error
        if radius**2 - (getDistance(arcintersect1, arcintersect2)**2) / 4 < 0:
            print("Error in getArcFacetWithIntersects({}, {}, {}, {}, {})".format(poly, area, epsilon, arcintersect1, arcintersect2))
            return None, None, None
        center = getCenter(arcintersect1, arcintersect2, radius)
        cura, intersects = getCircleIntersectArea(center, radius, poly)
        returnintersects = [facetintersect for _,facetintersect in sorted(zip(list(map(lambda point: getDistance(arcintersect1, point), intersects)), intersects))]
    if abs(cura - area)/polyarea < epsilon:
        return center, radius, returnintersects
    print("Failed to converge in getArcFacetWithIntersects({}, {}, {}, {}, {})".format(poly, area, epsilon, arcintersect1, arcintersect2))
    return None, None, None