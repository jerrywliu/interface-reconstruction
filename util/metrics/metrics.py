import math

from main.geoms.geoms import getDistance
from main.structs.facets.base_facet import LinearFacet, ArcFacet, CornerFacet, lerp, getNormal
from typing import Union


def hausdorffFacets(facet1: Union[LinearFacet, ArcFacet, CornerFacet], facet2: Union[LinearFacet, ArcFacet, CornerFacet], n=100):
    """
    Compute Hausdorff distance between two facets using numerical approximation.
    
    Args:
        facet1: First facet (LinearFacet, ArcFacet, or CornerFacet)
        facet2: Second facet (LinearFacet, ArcFacet, or CornerFacet)
        n: Number of equispaced points to sample along each facet (default: 100)
        
    Returns:
        float: Hausdorff distance
    """
    
    def sample_points_linear(facet, n):
        """Sample n equispaced points along a linear facet."""
        points = []
        for i in range(n):
            t = i / (n - 1)  # Parameter from 0 to 1
            point = lerp(facet.pLeft, facet.pRight, t)
            points.append(point)
        return points
    
    def sample_points_arc(facet, n):
        """Sample n equispaced points along an arc facet."""
        points = []
        
        # Calculate angles for start and end points
        start_angle = math.atan2(facet.pLeft[1] - facet.center[1], 
                                facet.pLeft[0] - facet.center[0])
        end_angle = math.atan2(facet.pRight[1] - facet.center[1], 
                              facet.pRight[0] - facet.center[0])
        
        # Handle angle wrapping - ensure we go the shorter way around
        angle_diff = end_angle - start_angle
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        # Sample points along the arc
        for i in range(n):
            t = i / (n - 1)  # Parameter from 0 to 1
            angle = start_angle + t * angle_diff
            point = [
                facet.center[0] + abs(facet.radius) * math.cos(angle),
                facet.center[1] + abs(facet.radius) * math.sin(angle)
            ]
            points.append(point)
        
        return points
    
    def sample_points_corner(facet, n):
        """Sample n equispaced points along a corner facet (both sides)."""
        points = []
        
        # Sample points along left facet
        if isinstance(facet.facetLeft, LinearFacet):
            left_points = sample_points_linear(facet.facetLeft, n // 2)
        elif isinstance(facet.facetLeft, ArcFacet):
            left_points = sample_points_arc(facet.facetLeft, n // 2)
        else:
            left_points = []
        
        # Sample points along right facet
        if isinstance(facet.facetRight, LinearFacet):
            right_points = sample_points_linear(facet.facetRight, n // 2)
        elif isinstance(facet.facetRight, ArcFacet):
            right_points = sample_points_arc(facet.facetRight, n // 2)
        else:
            right_points = []
        
        # Combine points, avoiding duplicate corner point
        if left_points and right_points:
            points = left_points[:-1] + right_points[1:]
        else:
            points = left_points + right_points
        return points
    
    def sample_points(facet, n):
        """Sample n points along a facet based on its type."""
        if isinstance(facet, LinearFacet):
            return sample_points_linear(facet, n)
        elif isinstance(facet, ArcFacet):
            return sample_points_arc(facet, n)
        elif isinstance(facet, CornerFacet):
            return sample_points_corner(facet, n)
        else:
            raise ValueError(f"Unsupported facet type: {type(facet)}")
    
    # Sample points from both facets
    points1 = sample_points(facet1, n)
    points2 = sample_points(facet2, n)
    
    # Compute Hausdorff distance: max(min distances from each point to other set)
    def compute_hausdorff_distance(set1, set2):
        """Compute Hausdorff distance from set1 to set2."""
        max_min_distance = 0
        for p1 in set1:
            min_distance = float('inf')
            for p2 in set2:
                dist = getDistance(p1, p2)
                min_distance = min(min_distance, dist)
            max_min_distance = max(max_min_distance, min_distance)
        return max_min_distance
    
    # Hausdorff distance is the maximum of both directions
    d1_to_2 = compute_hausdorff_distance(points1, points2)
    d2_to_1 = compute_hausdorff_distance(points2, points1)
    
    return max(d1_to_2, d2_to_1)


def calculate_facet_gaps(mesh, reconstructed_facets):
    """Calculate the minimum distance between left facet endpoint and neighboring facets' endpoints.

    Args:
        mesh: MergeMesh object
        reconstructed_facets: List of reconstructed facets for each cell

    Returns:
        avg_gap: Average minimum gap distance across all mixed cells
    """
    total_gap = 0
    cnt_gap = 0

    # Get list of merged polygons for easier indexing
    merged_polys = list(mesh.merged_polys.values())

    for i, (poly, facet) in enumerate(zip(merged_polys, reconstructed_facets)):
        if not facet:  # Skip if no facet
            continue

        # Get left endpoint of current facet
        left_endpoint = facet.pLeft

        # Find all neighboring mixed cells that have facets
        min_gap = float("inf")
        for neighbor in poly.adjacent_polys:
            if neighbor in merged_polys:
                # Get neighbor facet
                i = merged_polys.index(neighbor)
                neighbor_facet = reconstructed_facets[i]
                assert neighbor_facet == neighbor.getFacet()
                # Calculate distance to both endpoints of neighbor's facet
                dist_left = getDistance(left_endpoint, neighbor_facet.pLeft)
                dist_right = getDistance(left_endpoint, neighbor_facet.pRight)
                # Update minimum gap
                min_gap = min(min_gap, dist_left, dist_right)

        if min_gap != float("inf"):
            total_gap += min_gap
            cnt_gap += 1

    return total_gap / cnt_gap if cnt_gap > 0 else 0
