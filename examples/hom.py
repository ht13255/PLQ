"""Analytic HOM visibility with variable wavepacket overlap and propagation loss."""
from plq import Circuit, wavepacket_input

for overlap in [0.0,0.5,1.0]:
    photons=wavepacket_input(2,[0,1],[[1,overlap],[overlap,1]])
    result=photons.through(Circuit(2).bs(0,1).loss(0,.9).loss(1,.9))
    coincidence=result.spatial_probabilities().get((1,1),0)
    print(f"overlap={overlap:.1f}, coincidence={coincidence:.8f}, analytic={.9**2*(1-overlap**2)/2:.8f}")
