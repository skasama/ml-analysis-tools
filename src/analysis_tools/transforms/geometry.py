import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import as_vector_array


def normalize_vectors(
    vectors: ArrayLike,
) -> NDArray[np.float64]:
    """Normalize vectors to unit length."""
    vectors = as_vector_array(
        vectors,
        size=3,
        name="vectors",
    ).astype(float)

    norms = np.linalg.norm(vectors, axis=1)

    if np.any(norms == 0):
        raise ValueError("Cannot normalize zero-length vectors.")

    return vectors / norms[:, None]


def cylindrical_radius(
    positions: ArrayLike,
    *,
    axis: int = 2,
) -> NDArray[np.float64]:
    """

    Calculate cylindrical radius sqrt(x^2 + y^2).

    Parameters
    ----------
    positions: ArrayLike
        An array of shape (N, 3) representing N points in 3D space.
    axis: int
        The axis along which the cylinder is oriented (0, 1, or 2).

    Returns
    -------
    NDArray[np.float64]
        An array of shape (N,) representing the cylindrical radii of the points.
    """
    _validate_axis(axis)

    positions = as_vector_array(
        positions,
        size=3,
        name="positions",
    ).astype(float)

    transverse = np.delete(
        positions,
        axis,
        axis=1,
    )

    return np.linalg.norm(transverse, axis=1)


def compute_wall(
    positions: ArrayLike,
    *,
    radius: float,
    half_height: float,
    axis: int = 2,
) -> NDArray[np.float64]:
    """

    Calculate the shortest signed distance to a cylidrical wall.

    Positive values are inside the cylinder,
    Negative values are outside the cylinder.

    Parameters
    ----------
    positions: ArrayLike
        An array of shape (N, 3) representing N points in 3D space
    radius: float
        The radius of the cylinder
    half_height: float
        Half the height of the cylinder (the cylinder extends from -half_height
        to +half_height along the z-axis)
    axis: int
        The axis along which the cylinder is oriented (0, 1, or 2).

    Returns
    -------
    NDArray[np.float64]
        An array of shape (N,) representing the shortest signed distance to
        the cylinder wall for each point. Positive values indicate points inside
        the cylinder, while negative values indicate points outside the cylinder.
    """
    _validate_cylinder(radius, half_height)
    _validate_axis(axis)

    positions = as_vector_array(
        positions,
        size=3,
        name="positions",
    ).astype(float)

    radial_distance = radius - cylindrical_radius(
        positions,
        axis=axis,
    )

    axial_position = positions[:, axis]
    axial_distance = half_height - np.abs(axial_position)

    return np.minimum(radial_distance, axial_distance)


def compute_towall(
    positions: ArrayLike,
    directions: ArrayLike,
    *,
    radius: float,
    half_height: float,
    axis: int = 2,
) -> NDArray[np.float64]:
    """Calculate the distance to the wall along a direction.

    Positive values are inside the cylinder,
    Negative values are outside the cylinder.

    Parameters
    ----------
    positions: ArrayLike
        An array of shape (N, 3) representing N points in 3D space
    directions: ArrayLike
        An array of shape (N, 3) representing N direction vectors in 3D space
    radius: float
        The radius of the cylinder
    half_height: float
        Half the height of the cylinder (the cylinder extends from -half_height
        to +half_height along the z-axis)
    axis: int
        The axis along which the cylinder is oriented (0, 1, or 2).

    Returns
    -------
    NDArray[np.float64]
        An array of shape (N,) representing the distance to the cylinder wall
        along the given direction for each point. Positive values indicate points
        inside the cylinder, while negative values indicate points outside the cylinder.
    """
    _validate_cylinder(radius, half_height)
    _validate_axis(axis)

    positions = as_vector_array(
        positions,
        size=3,
        name="positions",
    ).astype(float)
    directions = normalize_vectors(directions)

    if len(positions) != len(directions):
        raise ValueError("positions and directions must have the same length.")

    wall = compute_wall(
        positions,
        radius=radius,
        half_height=half_height,
        axis=axis,
    )

    if np.any(wall < 0.0):
        raise ValueError("ToWall is undefined for positions outside the cylinder.")

    transverse_positions = np.delete(
        positions,
        axis,
        axis=1,
    )
    transverse_directions = np.delete(
        directions,
        axis,
        axis=1,
    )

    axial_positions = positions[:, axis]
    axial_directions = directions[:, axis]

    # Compute the distance from each point inside a finite cylinder to the first
    # cylinder boundary encountered along the direction vector (dx, dy, dz).
    #
    # The cylinder is centered on the z-axis, has radius `radius`,
    # and extends from z = -half_height to z = +half_height.
    #
    # First, find the intersection with the cylindrical side wall by solving
    #
    #     (x + t*dx)^2 + (y + t*dy)^2 = radius^2
    #
    # which yields a quadratic equation:
    #
    #     a*t^2 + b*t + c = 0
    #
    # where:
    #     a = dx^2 + dy^2
    #     b = 2(x*dx + y*dy)
    #     c = x^2 + y^2 - radius^2
    #
    # Only solutions with a non-negative discriminant correspond to valid
    # side-wall intersections. For points inside the cylinder, the forward
    # intersection is given by:
    #
    #     t = (-b + sqrt(discriminant)) / (2a)
    #
    # Next, compute the intersection with the top or bottom cap by solving:
    #
    #     z + t*dz = +half_height   (if dz > 0)
    #     z + t*dz = -half_height   (if dz < 0)
    #
    # The distance to the cylinder boundary is the smaller of:
    #     - the side-wall intersection distance
    #     - the cap intersection distance
    #
    # A negative distance indicates that the starting position lies outside
    # the cylinder, in which case the result is undefined.
    radial_a = np.sum(
        transverse_directions**2,
        axis=1,
    )
    radial_b = 2.0 * np.sum(
        transverse_positions * transverse_directions,
        axis=1,
    )
    radial_c = (
        np.sum(
            transverse_positions**2,
            axis=1,
        )
        - radius**2
    )

    discriminant = radial_b**2 - 4.0 * radial_a * radial_c

    radial_distance = np.full(len(positions), np.inf)
    valid_radial = (radial_a > 0.0) & (discriminant >= 0.0)

    radial_distance[valid_radial] = (
        -radial_b[valid_radial] + np.sqrt(discriminant[valid_radial])
    ) / (2.0 * radial_a[valid_radial])

    axial_distance = np.full(
        len(positions),
        np.inf,
        dtype=float,
    )

    positive_axial = axial_directions > 0.0
    negative_axial = axial_directions < 0.0

    axial_distance[positive_axial] = (
        half_height - axial_positions[positive_axial]
    ) / axial_directions[positive_axial]

    axial_distance[negative_axial] = (
        -half_height - axial_positions[negative_axial]
    ) / axial_directions[negative_axial]

    distances = np.minimum(radial_distance, axial_distance)

    if np.any(~np.isfinite(distances)):
        raise ValueError("Could not determine a finite ToWall distance.")

    if np.any(distances < 0.0):
        raise ValueError("ToWall is undefined for positions outside the cylinder.")

    return distances


def _validate_cylinder(
    radius: float,
    half_height: float,
) -> None:
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    if half_height <= 0.0:
        raise ValueError("half_height must be positive.")


def _validate_axis(axis: int) -> None:
    if axis not in {0, 1, 2}:
        raise ValueError("axis must be 0, 1, or 2.")


def _validate_cylinder(
    radius: float,
    half_height: float,
) -> None:
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
