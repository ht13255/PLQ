"""Thermal attenuation with an explicit, converging bath-tail error budget."""
from plq import Circuit

mean, transmission = .2, .7
for cutoff in (4, 8, 12):
    result = Circuit(1).thermal_loss(0, transmission, mean, bath_cutoff=cutoff).run([0])
    vacuum = result.probabilities()[(0,)]
    analytic_vacuum = 1/(1+(1-transmission)*mean)
    error = abs(vacuum-analytic_vacuum)
    print(f"bath_cutoff={cutoff:2d}  retained={result.state.trace:.12f}  "
          f"omitted={result.bath_omitted_probability:.3e}  vacuum_error={error:.3e}")
    assert error <= result.bath_omitted_probability+1e-13

# Or choose the cutoff from a requested bath tail (not experimental accuracy).
result = Circuit(1).thermal_loss(0, transmission, mean, tail_tolerance=1e-13).run([0])
print("automatic cutoff:", result.truncation_history)
