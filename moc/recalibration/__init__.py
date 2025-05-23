from .hdr_recalibrator import HDRRecalibrator
from .latent_recalibrator import LatentRecalibrator

recalibrators = {
    'HDR': HDRRecalibrator,
    'Latent': LatentRecalibrator,
}
