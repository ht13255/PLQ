"""Physical heralds -> optical output -> diagnostic small-code recovery."""
import numpy as np
from plq import (teleportation_instrument, FeedForward, Detector,
                 LogicalQPU, repetition_code)

# Illustrative assumptions, not parameters calibrated to a particular device.
hardware = teleportation_instrument(
    transmissions=[.95]*4+[.98]*2,
    detectors=[Detector(efficiency=.9, dark_rate_hz=100)]*4,
    schedule=FeedForward(detection_seconds=2e-9, processing_seconds=8e-9,
                         switching_seconds=2e-9, buffer_seconds=15e-9,
                         buffer_loss_db_per_second=2e6, switch_transmission=.97))
result = hardware.apply(np.eye(2)/2)
print({k: v for k, v in result.items() if k.endswith('_probability')})
print('computational probability:', result['computational_branch'].probability)
print('physical classical flags + retained optical output:', hardware.flagged_channel().completeness_residual)
# Diagnostic projection only: this does not implement a QND photon-presence
# detector. Ideal recovery below is explicitly a remaining hardware gap.
qpu = LogicalQPU(repetition_code(3)).prepare(np.array([1, 1j])/np.sqrt(2))
for wire in range(3):
    qpu.local_channel(hardware.computational_channel(), [wire])
logical = qpu.correct().logical_state()
print('absolute logical branch weight:', logical.probability)
print('conditional logical density:', logical.conditional())
