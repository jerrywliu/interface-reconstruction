from main.geoms.geoms import getDistance


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
