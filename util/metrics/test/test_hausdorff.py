#!/usr/bin/env python3
"""
Test script for the hausdorffFacets function.
Tests the numerical Hausdorff distance calculation between different facet types.
"""

from main.structs.facets.base_facet import (
    LinearFacet, 
    ArcFacet, 
    CornerFacet,
)
from util.metrics.metrics import hausdorffFacets


def test_linear_facets():
    """Test Hausdorff distance between linear facets."""
    print("Testing linear facets...")
    
    # Create two linear facets
    facet1 = LinearFacet([0, 0], [1, 0])  # Horizontal line
    facet2 = LinearFacet([0, 1], [1, 1])  # Parallel horizontal line
    
    distance = hausdorffFacets(facet1, facet2)
    print(f"Distance: {distance:.6f}")
    print(f"Expected: 1.0")
    print(f"Difference: {abs(distance - 1.0):.6f}")
    print()


def test_arc_facets():
    """Test Hausdorff distance between arc facets."""
    print("Testing arc facets...")
    
    # Create two arc facets with same center but different radii
    center = [0, 0]
    facet1 = ArcFacet(center, 1.0, [1, 0], [0, 1])  # Quarter circle, radius 1
    facet2 = ArcFacet(center, 2.0, [2, 0], [0, 2])  # Quarter circle, radius 2
    
    distance = hausdorffFacets(facet1, facet2)
    print(f"Distance: {distance:.6f}")
    print(f"Expected: 1.0 (difference in radii)")
    print(f"Difference: {abs(distance - 1.0):.6f}")
    print()


def test_corner_facets():
    """Test Hausdorff distance between corner facets."""
    print("Testing corner facets...")
    
    # Create corner facets with linear sides
    facet1 = CornerFacet(
        centerLeft=None, centerRight=None,
        radiusLeft=None, radiusRight=None,
        pLeft=[0, 0], corner=[1, 0], pRight=[1, 1]
    )
    
    facet2 = CornerFacet(
        centerLeft=None, centerRight=None,
        radiusLeft=None, radiusRight=None,
        pLeft=[0, 1], corner=[1, 1], pRight=[1, 2]
    )
    
    distance = hausdorffFacets(facet1, facet2)
    print(f"Distance: {distance:.6f}")
    print(f"Expected: 1.0 (vertical separation)")
    print(f"Difference: {abs(distance - 1.0):.6f}")
    print()


def test_mixed_facets():
    """Test Hausdorff distance between different facet types."""
    print("Testing mixed facet types...")
    
    # Linear vs Arc
    linear_facet = LinearFacet([0, 0], [1, 0])
    arc_facet = ArcFacet([0.5, 0], 0.5, [0, 0], [1, 0])
    
    distance = hausdorffFacets(linear_facet, arc_facet)
    print(f"Linear vs Arc distance: {distance:.6f}")
    
    # Linear vs Corner
    corner_facet = CornerFacet(
        centerLeft=None, centerRight=None,
        radiusLeft=None, radiusRight=None,
        pLeft=[0, 0], corner=[0.5, 0], pRight=[0.5, 0.5]
    )
    
    distance2 = hausdorffFacets(linear_facet, corner_facet)
    print(f"Linear vs Corner distance: {distance2:.6f}")
    print()


def test_convergence():
    """Test convergence of numerical method with increasing n."""
    print("Testing convergence...")
    
    facet1 = LinearFacet([0, 0], [1, 0])
    facet2 = LinearFacet([0, 1], [1, 1])
    
    for n in [10, 20, 50, 100, 200]:
        distance = hausdorffFacets(facet1, facet2, n=n)
        print(f"n={n:3d}: distance={distance:.6f}")
    
    print()


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")
    
    # Identical facets
    facet1 = LinearFacet([0, 0], [1, 0])
    facet2 = LinearFacet([0, 0], [1, 0])
    
    distance = hausdorffFacets(facet1, facet2)
    print(f"Identical facets distance: {distance:.6f}")
    
    # Very small facets
    facet3 = LinearFacet([0, 0], [0.001, 0])
    facet4 = LinearFacet([0, 0.001], [0.001, 0.001])
    
    distance2 = hausdorffFacets(facet3, facet4)
    print(f"Small facets distance: {distance2:.6f}")
    
    print()


def test_problematic_cases():
    """Test cases that previously showed discrepancies with closed-form method."""
    print("Testing previously problematic cases...")
    
    # Test case 1: Diagonal lines
    facet1 = LinearFacet([0, 0], [1, 1])
    facet2 = LinearFacet([0, 1], [1, 2])
    
    distance = hausdorffFacets(facet1, facet2)
    print(f"Diagonal lines distance: {distance:.6f}")
    
    # Test case 2: Different lengths
    facet3 = LinearFacet([0, 0], [2, 0])
    facet4 = LinearFacet([0, 1], [1, 1])
    
    distance2 = hausdorffFacets(facet3, facet4)
    print(f"Different lengths distance: {distance2:.6f}")
    
    # Test case 3: Interior point maximum
    facet5 = LinearFacet([0, 0], [2, 0])
    facet6 = LinearFacet([0, 1], [1, 1])
    
    distance3 = hausdorffFacets(facet5, facet6)
    print(f"Interior point maximum distance: {distance3:.6f}")
    print(f"Expected: ~1.414 (√2)")
    print(f"Difference: {abs(distance3 - 1.414):.6f}")
    
    print()


if __name__ == "__main__":
    print("Testing Hausdorff distance function")
    print("=" * 50)
    
    test_linear_facets()
    test_arc_facets()
    test_corner_facets()
    test_mixed_facets()
    test_convergence()
    test_edge_cases()
    test_problematic_cases()
    
    print("All tests completed!") 