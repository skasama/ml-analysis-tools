from enum import IntEnum


class Particle(IntEnum):
    """

    Particle labels used in the analysis datasets. (tentative)

    Attributes
    ----------
    MUON : int(default=0)
        Label for muon particle.
    ELECTRON : int(default=1)
        Label for electron particle.
    CHARGED_PION : int(default=2)
        Label for charged pion particle.
    """

    MUON = 0
    ELECTRON = 1
    CHARGED_PION = 2


PARTICLE_MASS_MEV = {
    Particle.MUON: 105.65836668,
    Particle.ELECTRON: 0.510998950,
    Particle.CHARGED_PION: 139.57039,
}
