import numpy as np

EPSILON = 1.0e-12


def prepare_xyz(values, name="values"):
    """Convert an array to shape (N, 3)."""
    values = np.asarray(values, dtype=float)

    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), but received {values.shape}.")

    return values


def cylindrical_radius(vertices):
    """Return the cylindrical radius R = sqrt(x^2 + y^2)."""
    vertices = prepare_xyz(
        vertices,
        "vertices",
    )

    return np.hypot(
        vertices[:, 0],
        vertices[:, 1],
    )


def unit(vectors):
    """Normalize vectors to unit length."""
    vectors = prepare_xyz(
        vectors,
        "vectors",
    )
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= EPSILON):
        n_invalid = int(np.sum(norms[:, 0] <= EPSILON))
        raise ValueError(f"Found {n_invalid} Zero-length direction vectors.")

    return vectors / norms


def opening_angle(true, pred):
    """Return the opening angle in degrees."""
    true = prepare_xyz(
        true,
        "true",
    )
    pred = prepare_xyz(
        pred,
        "pred",
    )

    if len(true) != len(pred):
        raise ValueError("true and pred must have the same number of events.")

    true_unit = unit(true)
    pred_unit = unit(pred)

    cosine = np.sum(
        true_unit * pred_unit,
        axis=1,
    )

    return np.degrees(
        np.arccos(
            np.clip(
                cosine,
                -1.0,
                1.0,
            )
        )
    )


def wall(
    vertices,
    radius=1690.0,
    half_height=1810.0,
):
    """Return signed distance to the nearest cylindrical boundary.

    Positive values indicate vertices inside the cylinder.
    Negative values indicate vertices outside the cylinder.
    """
    vertices = prepare_xyz(
        vertices,
        "vertices",
    )

    radial_gap = radius - cylindrical_radius(vertices)
    vertical_gap = half_height - np.abs(vertices[:, 2])

    inside = (radial_gap >= 0.0) & (vertical_gap >= 0.0)

    distance = np.minimum(
        np.abs(radial_gap),
        np.abs(vertical_gap),
    )

    return np.where(
        inside,
        distance,
        -distance,
    )


def towall(
    vertices,
    directions,
    radius=1690.0,
    half_height=1810.0,
):
    """Return the forward distance to a finite cylindrical boundary."""
    vertices = prepare_xyz(
        vertices,
        "vertices",
    )
    directions = prepare_xyz(
        directions,
        "directions",
    )

    if len(vertices) != len(directions):
        raise ValueError("vertices and directions must have the same number of events.")

    directions = unit(directions)

    x, y, z = vertices.T
    dx, dy, dz = directions.T

    candidates = np.full(
        (len(vertices), 4),
        np.inf,
    )

    # Intersections with the cylindrical side wall.
    a = dx**2 + dy**2
    b = 2.0 * (x * dx + y * dy)
    c = x**2 + y**2 - radius**2

    discriminant = b**2 - 4.0 * a * c

    valid_quadratic = (a > EPSILON) & (discriminant >= 0.0)

    square_root = np.zeros_like(discriminant)
    square_root[valid_quadratic] = np.sqrt(discriminant[valid_quadratic])

    side_distances = [
        np.full(len(vertices), np.inf),
        np.full(len(vertices), np.inf),
    ]

    side_distances[0][valid_quadratic] = (
        -b[valid_quadratic] + square_root[valid_quadratic]
    ) / (2.0 * a[valid_quadratic])

    side_distances[1][valid_quadratic] = (
        -b[valid_quadratic] - square_root[valid_quadratic]
    ) / (2.0 * a[valid_quadratic])

    for candidate_index, distance in enumerate(side_distances):
        intersection_z = z + distance * dz

        valid_side = (
            np.isfinite(distance)
            & (distance > EPSILON)
            & (np.abs(intersection_z) <= half_height)
        )

        candidates[valid_side, candidate_index] = distance[valid_side]

    # Intersections with the top and bottom caps.
    valid_z = np.abs(dz) > EPSILON

    top_distance = np.full(
        len(vertices),
        np.inf,
    )
    bottom_distance = np.full(
        len(vertices),
        np.inf,
    )

    top_distance[valid_z] = (half_height - z[valid_z]) / dz[valid_z]

    bottom_distance[valid_z] = (-half_height - z[valid_z]) / dz[valid_z]

    cap_distances = [
        top_distance,
        bottom_distance,
    ]

    for offset, distance in enumerate(
        cap_distances,
        start=2,
    ):
        intersection_x = x + distance * dx
        intersection_y = y + distance * dy

        intersection_radius = np.hypot(
            intersection_x,
            intersection_y,
        )

        valid_cap = (
            np.isfinite(distance)
            & (distance > EPSILON)
            & (intersection_radius <= radius)
        )

        candidates[valid_cap, offset] = distance[valid_cap]

    distances = np.min(
        candidates,
        axis=1,
    )

    if np.any(~np.isfinite(distances)):
        n_invalid = int(np.sum(~np.isfinite(distances)))
        raise ValueError(f"Could not calculate ToWall for {n_invalid} events.")

    return distances
