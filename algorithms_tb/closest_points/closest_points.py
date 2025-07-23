"""
Calculate closest points between two line segments in 3D space.
--------------------------
author: Matthias Gassilloud
date: 05.06.2025
--------------------------

The method was adapted from Stack Overflow [1] and optimized with assistance
from the Claude 3.7 Sonnet large language model (Anthropic, 2024).


References:

[1] Stack Overflow. "Shortest distance between two line segments." 
    https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
    
"""

import numpy as np
from numba import jit
from typing import Tuple
import matplotlib.pyplot as plt


@jit(nopython=True, fastmath=True, cache=True, error_model='numpy')
def closest_points_between_lines(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    clampAll: bool = True,
    clampA0: bool = False,
    clampA1: bool = False,
    clampB0: bool = False,
    clampB1: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Find closest points between two 3D line segments.
    
    Parameters:
    -----------
    a0, a1 : np.ndarray - First line segment endpoints
    b0, b1 : np.ndarray - Second line segment endpoints
    clampAll : bool - If True, clamp all endpoints (default)
    clampA0, clampA1 : bool - Clamp start/end of first line
    clampB0, clampB1 : bool - Clamp start/end of second line
    
    Returns:
    --------
    point_on_segment_a : np.ndarray - Closest point on first segment
    point_on_segment_b : np.ndarray - Closest point on second segment
    distance : float - Distance between closest points
    
    Notes:
    ------
    - Returns np.nan for parallel segments with no unique solution
    - Uses a threshold for parallel detection for numerical stability
    """
    
    # Input validation
    for p in (a0, a1, b0, b1):
        if p.size != 3:
            raise ValueError("All points must be 3D vectors")
            
    # Configure clamping flags
    if clampAll:
        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

    # Calculate segment vectors and their magnitudes
    A = a1 - a0  # Direction vector of first segment
    B = b1 - b0  # Direction vector of second segment
    
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    
    # Handle degenerate cases: zero-length segments
    # A zero-length segment is a point, so we find closest point from point to segment
    if magA < 1e-10:
        if magB < 1e-10:
            # Both segments are points - simple point-to-point distance
            return a0, b0, np.linalg.norm(a0 - b0)
        
        # Avoid redundant calculations by computing closest point once
        closest_on_b = closest_point_between_point_and_segment(a0, b0, b1, clampB0, clampB1)
        return a0, closest_on_b, np.linalg.norm(a0 - closest_on_b)
    
    if magB < 1e-10:
        # Same for other segment
        closest_on_a = closest_point_between_point_and_segment(b0, a0, a1, clampA0, clampA1)
        return closest_on_a, b0, np.linalg.norm(closest_on_a - b0)

    # Normalize direction vectors for numerical stability
    _A = A / magA
    _B = B / magB
    
    # Calculate cross product to check for parallel lines
    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross)**2
    
    # Handle parallel lines - threshold to capture near-parallel cases
    # where floating-point precision might cause issues
    if denom < 1e-10:
        return closest_distance_parallel_segments(
            a0, a1, b0, b1, _A, magA,
            clampA0, clampA1, clampB0, clampB1
        )
    
    # Lines are not parallel - find closest points using standard method
    
    # Calculate vector between origins
    t = b0 - a0
    
    # Use determinants to find parameters for closest points
    detA = np.linalg.det(np.vstack((t, _B, cross)))
    detB = np.linalg.det(np.vstack((t, _A, cross)))
    
    # Calculate parameters along each segment
    t0 = detA / denom  # parameter for first segment
    t1 = detB / denom  # parameter for second segment

    # Check if clamping is required
    needs_clamping = ((clampA0 and t0 < 0) or 
                      (clampA1 and t0 > magA) or
                      (clampB0 and t1 < 0) or
                      (clampB1 and t1 > magB))
    
    if needs_clamping:
        # Apply clamping and handle subsequent adjustments
        pA, pB = clamp_segment_points(
            a0, _A, magA, b0, _B, magB, t0, t1,
            clampA0, clampA1, clampB0, clampB1
        )
    else:
        pA = a0 + (_A * t0)  # use unscaled parameters with normalized vectors
        pB = b0 + (_B * t1)  # use unscaled parameters with normalized vectors
    
    # Calculate final distance
    distance = np.linalg.norm(pA - pB)
    
    return pA, pB, distance


@jit(nopython=True, fastmath=True, cache=True, error_model='numpy')
def closest_distance_parallel_segments(
    a0, a1, b0, b1, _A, magA,
    clampA0, clampA1, clampB0, clampB1
):
    """
    Handle the special case of parallel line segments to find closest points.
    
    Parameters:
    -----------
    a0 : np.ndarray - Start point of first segment
    a1 : np.ndarray - End point of first segment
    b0 : np.ndarray - Start point of second segment
    b1 : np.ndarray - End point of second segment
    _A : np.ndarray - Normalized direction vector of first segment
    magA : float - Length of first segment
    clampA0 : bool - Whether to clamp at start of first segment
    clampA1 : bool - Whether to clamp at end of first segment
    clampB0 : bool - Whether to clamp at start of second segment
    clampB1 : bool - Whether to clamp at end of second segment
    
    Returns:
    --------
    point_on_segment_a : np.ndarray - Closest point on first segment (NaN for true parallel)
    point_on_segment_b : np.ndarray - Closest point on second segment (NaN for true parallel)
    distance : float - Perpendicular distance between the parallel segments
    
    Notes:
    ------
    For truly parallel segments without unique closest points, returns NaN coordinates
    but always computes the perpendicular distance correctly.
    Paramteters _B and magB are not used in this function since all calculations can be 
    performed using only the first lines direction vector (_A) when the lines are parallel.
    """
    # Project b0 onto line A to determine relative positions
    d0 = np.dot(_A, (b0 - a0))
    
    # If we have clamping constraints, check for endpoint solutions
    if clampA0 or clampA1 or clampB0 or clampB1:
        d1 = np.dot(_A, (b1 - a0))
        
        # Case 1: Segment B is before segment A
        if d0 <= 0 >= d1:
            if clampA0 and clampB1:
                # Choose the closer pair of endpoints
                if np.absolute(d0) < np.absolute(d1):
                    return a0, b0, np.linalg.norm(a0 - b0)
                return a0, b1, np.linalg.norm(a0 - b1)
        
        # Case 2: Segment B is after segment A
        elif d0 >= magA <= d1:
            if clampA1 and clampB0:
                # Choose the closer pair of endpoints
                if np.absolute(d0 - magA) < np.absolute(d1 - magA):
                    return a1, b0, np.linalg.norm(a1 - b0)
                return a1, b1, np.linalg.norm(a1 - b1)
    
    # Calculate perpendicular distance between parallel lines
    v = b0 - a0
    perpDist = np.linalg.norm(v - np.dot(v, _A) * _A)
    
    # For parallel lines without a unique solution, return NaN points
    return np.full(3, np.nan), np.full(3, np.nan), perpDist


@jit(nopython=True, fastmath=True, cache=True, error_model='numpy')
def clamp_segment_points(
    a0, _A, magA, b0, _B, magB, t0, t1,
    clampA0, clampA1, clampB0, clampB1
):
    """
    Clamp points to line segments according to constraints and recalculate their positions.
    
    Parameters:
    -----------
    a0 : np.ndarray - Start point of first segment
    _A : np.ndarray - Normalized direction vector of first segment
    magA : float - Length of first segment
    b0 : np.ndarray - Start point of second segment
    _B : np.ndarray - Normalized direction vector of second segment
    magB : float - Length of second segment
    t0 : float - Parameter value for point on first segment (0 to magA)
    t1 : float - Parameter value for point on second segment (0 to magB)
    clampA0 : bool - Whether to clamp at start of first segment
    clampA1 : bool - Whether to clamp at end of first segment
    clampB0 : bool - Whether to clamp at start of second segment
    clampB1 : bool - Whether to clamp at end of second segment
    
    Returns:
    --------
    pA : np.ndarray - Clamped point on first segment
    pB : np.ndarray - Clamped point on second segment
    
    Notes:
    ------
    When a point is clamped on one segment, this function recalculates the corresponding 
    point on the other segment to maintain minimum distance between segments.
    """
    # Initialize points at unclamped positions
    pA = a0 + (_A * t0)
    pB = b0 + (_B * t1)
    
    # Clamp first segment point if needed
    if clampA0 and t0 < 0:
        pA = a0  # Clamp to start point
    elif clampA1 and t0 > magA:
        pA = a0 + _A * magA  # Clamp to end point
    
    # Clamp second segment point if needed
    if clampB0 and t1 < 0:
        pB = b0  # Clamp to start point
    elif clampB1 and t1 > magB:
        pB = b0 + _B * magB  # Clamp to end point
    
    # If first point was clamped, recalculate second point
    if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
        # Project clamped point pA onto segment B
        dot = np.dot(_B, (pA - b0))
        
        # Clamp projection if needed
        if clampB0 and dot < 0:
            dot = 0
        elif clampB1 and dot > magB:
            dot = magB
            
        # Update second point based on projection
        pB = b0 + (_B * dot)
    
    # If second point was clamped, recalculate first point
    if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
        # Project clamped point pB onto segment A
        dot = np.dot(_A, (pB - a0))
        
        # Clamp projection if needed
        if clampA0 and dot < 0:
            dot = 0
        elif clampA1 and dot > magA:
            dot = magA
            
        # Update first point based on projection
        pA = a0 + (_A * dot)
            
    return pA, pB


@jit(nopython=True, fastmath=True, cache=True, error_model='numpy')
def closest_point_between_point_and_segment(
    point, 
    segment_start, 
    segment_end,
    clamp_start=True,
    clamp_end=True
):
    """
    Find closest point on a line segment to a given point.
    
    Parameters:
    -----------
    point : np.ndarray - Point in 3D space
    segment_start : np.ndarray - Start of line segment
    segment_end : np.ndarray - End of line segment
    clamp_start : bool - Clamp at segment start
    clamp_end : bool - Clamp at segment end
        
    Returns:
    --------
    np.ndarray - Closest point on the line segment
    """
    segment_vec = segment_end - segment_start
    point_vec = point - segment_start
    
    segment_len = np.linalg.norm(segment_vec)
    
    # Handle degenerate case
    if segment_len < 1e-10:
        return segment_start
    
    # Normalize segment vector
    segment_vec_norm = segment_vec / segment_len
    
    # Project point onto segment line
    projection = np.dot(point_vec, segment_vec_norm)
    
    # Apply clamping constraints
    if clamp_start and projection < 0:
        projection = 0
    if clamp_end and projection > segment_len:
        projection = segment_len
    
    # Calculate closest point
    closest = segment_start + projection * segment_vec_norm
    
    return closest


def is_parallel_case(points):
    """Check if result indicates a parallel case (contains NaN)"""
    return np.any(np.isnan(points[0]))


def visualize_segments_3d(a0, a1, b0, b1, p1, p2, dist, title=None, subplot_position=None):
    """
    Visualize two line segments and their closest points in 3D.
    
    Parameters:
    -----------
    a0, a1 : np.ndarray - First line segment endpoints
    b0, b1 : np.ndarray - Second line segment endpoints
    p1 : np.ndarray - Closest point on first segment
    p2 : np.ndarray - Closest point on second segment
    dist : float - Distance between closest points
    title : str - Title for the plot
    subplot_position : tuple - Position in a subplot grid (ax parameter)
    """
    # Create figure if not using subplots - now with larger size for individual plots
    if subplot_position is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = subplot_position
    
    # Plot first line segment (red)
    ax.plot([a0[0], a1[0]], [a0[1], a1[1]], [a0[2], a1[2]], 'r-', linewidth=3, label='Segment A')
    
    # Plot second line segment (blue)
    ax.plot([b0[0], b1[0]], [b0[1], b1[1]], [b0[2], b1[2]], 'b-', linewidth=3, label='Segment B')
    
    # Mark segment endpoints with labels
    ax.scatter([a0[0]], [a0[1]], [a0[2]], color='darkred', s=100, marker='o')
    ax.scatter([a1[0]], [a1[1]], [a1[2]], color='red', s=100, marker='o')
    ax.scatter([b0[0]], [b0[1]], [b0[2]], color='darkblue', s=100, marker='o')
    ax.scatter([b1[0]], [b1[1]], [b1[2]], color='blue', s=100, marker='o')
    
    # Add endpoint labels
    ax.text(a0[0], a0[1], a0[2], '  a0', fontsize=12, color='darkred')
    ax.text(a1[0], a1[1], a1[2], '  a1', fontsize=12, color='red')
    ax.text(b0[0], b0[1], b0[2], '  b0', fontsize=12, color='darkblue')
    ax.text(b1[0], b1[1], b1[2], '  b1', fontsize=12, color='blue')
    
    # If not a parallel case, plot closest points and connecting line
    if not np.any(np.isnan(p1)):
        # Plot closest points
        ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                  color=['lime', 'forestgreen'], s=120, label='Closest Points')
        
        # Add labels for closest points
        ax.text(p1[0], p1[1], p1[2], '  p1', fontsize=12, color='lime')
        ax.text(p2[0], p2[1], p2[2], '  p2', fontsize=12, color='forestgreen')
        
        # Plot line between closest points
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'g--', linewidth=3)
        
        # Add distance annotation
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        ax.text(mid_x, mid_y, mid_z, f'  d={dist:.4f}', fontsize=14, color='green', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    else:
        # For parallel case, show distance with better visualization
        ax.plot([a0[0], b0[0]], [a0[1], b0[1]], [a0[2], b0[2]], 'g--', linewidth=2)
        
        # Add explanatory text for parallel case
        mid_x = (a0[0] + b0[0]) / 2
        mid_y = (a0[1] + b0[1]) / 2
        mid_z = (a0[2] + b0[2]) / 2
        ax.text(mid_x, mid_y, mid_z, f'Parallel Lines\nDistance = {dist:.4f}', fontsize=14, color='green',
               ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='green'))

    # Set labels and title with larger font
    ax.set_xlabel('X-axis', fontsize=14)
    ax.set_ylabel('Y-axis', fontsize=14)
    ax.set_zlabel('Z-axis', fontsize=14)
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Keep equal aspect ratio for better visualization
    # Find max range to make axis limits equal
    max_range = np.max([
        np.max([a0[0], a1[0], b0[0], b1[0]]) - np.min([a0[0], a1[0], b0[0], b1[0]]),
        np.max([a0[1], a1[1], b0[1], b1[1]]) - np.min([a0[1], a1[1], b0[1], b1[1]]),
        np.max([a0[2], a1[2], b0[2], b1[2]]) - np.min([a0[2], a1[2], b0[2], b1[2]])
    ])
    
    # Add a small buffer to ensure points aren't at the edge
    max_range = max_range * 1.2
    
    # Get the center point for each axis
    mid_x = (np.max([a0[0], a1[0], b0[0], b1[0]]) + np.min([a0[0], a1[0], b0[0], b1[0]])) / 2
    mid_y = (np.max([a0[1], a1[1], b0[1], b1[1]]) + np.min([a0[1], a1[1], b0[1], b1[1]])) / 2
    mid_z = (np.max([a0[2], a1[2], b0[2], b1[2]]) + np.min([a0[2], a1[2], b0[2], b1[2]])) / 2
    
    # Set limits to ensure equal scaling
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Add grid for better spatial reference
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add custom legend with distance information
    if not np.any(np.isnan(p1)):
        legend_elements = [
            plt.Line2D([0], [0], color='r', lw=3, label='Segment A'),
            plt.Line2D([0], [0], color='b', lw=3, label='Segment B'),
            plt.Line2D([0], [0], color='g', linestyle='--', lw=3, label=f'Shortest Distance: {dist:.4f}')
        ]
    else:
        legend_elements = [
            plt.Line2D([0], [0], color='r', lw=3, label='Segment A'),
            plt.Line2D([0], [0], color='b', lw=3, label='Segment B'),
            plt.Line2D([0], [0], color='g', linestyle='--', lw=3, label=f'Parallel Distance: {dist:.4f}')
        ]
        
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Only show the result if not in subplot
    if subplot_position is None:
        plt.tight_layout()
        plt.show()


def example_usage():
    """
    Example showing usage of closest points function with 3D visualizations.
    Each example will have its own standalone plot.
    """
    print("Basic closest points example:")
    a0 = np.array([0.0, 0.0, 0.0])
    a1 = np.array([1.0, 0.0, 0.0])
    
    b0 = np.array([0.5, 1.0, 0.5])
    b1 = np.array([1.0, 1.0, 1.0])
    
    p1, p2, dist = closest_points_between_lines(a0, a1, b0, b1)
    
    print("Point on first segment:", p1)
    print("Point on second segment:", p2)
    print("Distance:", dist)
    
    # Basic example visualization (individual plot)
    visualize_segments_3d(a0, a1, b0, b1, p1, p2, dist, 
                          title="Basic Example")
    
    print("\nCustom clamping example:")
    p1_custom, p2_custom, dist_custom = closest_points_between_lines(
        a0, a1, b0, b1, 
        clampAll=False, 
        clampA0=False, clampA1=True, 
        clampB0=False, clampB1=True
    )
    
    print("Point on first segment:", p1_custom)
    print("Point on second segment:", p2_custom)
    print("Distance:", dist_custom)
    
    # Custom clamping example visualization (individual plot)
    visualize_segments_3d(a0, a1, b0, b1, p1_custom, p2_custom, dist_custom, 
                          title="Custom Clamping")
    
    print("\nParallel segments example:")
    a0_p = np.array([0.0, 0.0, 0.0])
    a1_p = np.array([1.0, 0.0, 0.0])
    
    b0_p = np.array([0.0, 1.0, 0.0])
    b1_p = np.array([1.0, 1.0, 0.0])
    
    p1_p, p2_p, dist_p = closest_points_between_lines(a0_p, a1_p, b0_p, b1_p)
    
    print("Point on first segment:", p1_p)
    print("Point on second segment:", p2_p)
    print("Distance:", dist_p)
    print("Is parallel case:", is_parallel_case((p1_p, p2_p)))
    
    # Parallel segments example visualization (individual plot)
    visualize_segments_3d(a0_p, a1_p, b0_p, b1_p, p1_p, p2_p, dist_p, 
                          title="Parallel Segments")
    
    print("\nNon-orthogonal segments example:")
    # More interesting example with non-orthogonal segments
    a0_non_o = np.array([0.0, 0.0, 0.0])
    a1_non_o = np.array([2.0, 1.0, 0.5])
    
    b0_non_o = np.array([1.5, -0.5, 1.0])
    b1_non_o = np.array([0.5, 2.5, 0.7])
    
    p1_non_o, p2_non_o, dist_non_o = closest_points_between_lines(a0_non_o, a1_non_o, b0_non_o, b1_non_o)
    
    print("Point on first segment:", p1_non_o)
    print("Point on second segment:", p2_non_o)
    print("Distance:", dist_non_o)
    
    # Non-orthogonal segments example (individual plot)
    visualize_segments_3d(a0_non_o, a1_non_o, b0_non_o, b1_non_o, p1_non_o, p2_non_o, dist_non_o,
                          title="Non-Orthogonal Segments Example")



if __name__ == "__main__":
    example_usage()
