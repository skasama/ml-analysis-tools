import numpy as np

def cylindrical_radius(vertices):
    """Return the cylindrical radius R = sqrt(x^2 + y^2)."""
    vertices = np.asarray(vertices, dtype=float)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            "vertices must have shape (N, 3), "
            f"but received {vertices.shape}."
        )

    return np.hypot(
        vertices[:, 0],
        vertices[:, 1],
    )


def prepare_xyz(values, name="values"):
    """Convert an array to shape (N, 3)."""
    values = np.asarray(values, dtype=float)

    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"{name} must have shape (N, 3), "
            f"but received {values.shape}."
        )

    return values


def unit(vectors):
    """Normalize vectors to unit length."""
    vectors = prepare_xyz(vectors, "vectors")
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError(
            "Zero-length direction vector found."
        )

    return vectors / norms


def opening_angle(true, pred):
    """Return the opening angle in degrees."""
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


def cylindrical_radius(vertices):
    """Return R = sqrt(x^2 + y^2)."""
    vertices = prepare_xyz(
        vertices,
        "vertices",
    )

    return np.hypot(
        vertices[:, 0],
        vertices[:, 1],
    )


def wall(
    vertices,
    radius=1690.0,
    half_height=1810.0,
):
    """Return signed distance to the nearest cylindrical boundary."""
    vertices = np.asarray(vertices, dtype=float)

    radial_gap = radius - cylindrical_radius(vertices)
    vertical_gap = half_height - np.abs(vertices[:, 2])

    inside = (
        (radial_gap >= 0)
        & (vertical_gap >= 0)
    )

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
    """Return the forward distance to the cylindrical boundary."""
    vertices = prepare_xyz(
        vertices,
        "vertices",
    )
    directions = unit(directions)

    if len(vertices) != len(directions):
        raise ValueError(
            "vertices and directions must have "
            "the same number of events."
        )

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

    valid_side = (
        (a > 0)
        & (discriminant >= 0)
    )

    square_root = np.zeros_like(
        discriminant
    )
    square_root[valid_side] = np.sqrt(
        discriminant[valid_side]
    )

    candidates[valid_side, 0] = (
        -b[valid_side]
        + square_root[valid_side]
    ) / (2.0 * a[valid_side])

    candidates[valid_side, 1] = (
        -b[valid_side]
        - square_root[valid_side]
    ) / (2.0 * a[valid_side])

    # Intersections with the top and bottom caps.
    valid_z = np.abs(dz) > 0

    candidates[valid_z, 2] = (
        half_height - z[valid_z]
    ) / dz[valid_z]

    candidates[valid_z, 3] = (
        -half_height - z[valid_z]
    ) / dz[valid_z]

    # Only forward intersections are valid.
    candidates[candidates <= 0] = np.inf

    distances = np.min(
        candidates,
        axis=1,
    )

    if np.any(~np.isfinite(distances)):
        n_invalid = int(
            np.sum(~np.isfinite(distances))
        )
        raise ValueError(
            "Could not calculate ToWall for "
            f"{n_invalid} events."
        )

    return distances