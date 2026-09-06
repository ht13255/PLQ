"""Measured optical scaling and hardware regressions; no simulated speedup claims."""
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from time import perf_counter
import hashlib
import json
import platform
import sys
import numpy as np
from plq import Circuit, __version__, teleportation_instrument, FeedForward

ROOT = Path(__file__).resolve().parents[1]


def mesh(modes, photons, layers):
    circuit = Circuit(modes)
    for layer in range(layers):
        for m in range(layer % 2, modes-1, 2):
            circuit.bs(m, m+1, transmission=.43, phase=.17*layer)
    occ = [0]*modes
    for m in np.linspace(1, modes-2, photons, dtype=int):
        occ[m] += 1
    return circuit, occ


def main():
    cases=[]
    for modes, photons, layers, reference in ((6,3,3,True),(16,4,3,False)):
        circuit, occ = mesh(modes, photons, layers)
        start=perf_counter(); sparse=circuit.run_sparse(occ); elapsed=perf_counter()-start
        dimension=comb(modes+photons, photons)
        case={'modes':modes,'photons':photons,'layers':layers,'input_occupation':occ,
              'sparse_seconds':elapsed,'support_terms':len(sparse.amplitudes),'norm_squared':sparse.norm,
              'dense_total_cutoff_dimension':dimension,'single_dense_matrix_bytes_theoretical':16*dimension**2,
              'sparse_output_dictionary_bytes_measured':sys.getsizeof(sparse.amplitudes)+sum(sys.getsizeof(k)+sys.getsizeof(v) for k,v in sparse.amplitudes.items()),
              'memory_scope':'Shallow retained dict/tuple/complex sizes; excludes shared integer objects, mode matrices, interpreter and peak temporaries.'}
        if reference:
            start=perf_counter();dense=circuit.run(occ);case['dense_seconds']=perf_counter()-start
            case['max_density_error']=float(np.max(np.abs(sparse.to_fock().rho-dense.state.rho)))
            assert case['max_density_error'] < 1e-12
        else:
            case['dense_run']='Not allocated: exceeds default dense dimension budget. Bytes are theoretical, not measured RSS.'
        assert abs(sparse.norm-1) < 1e-12
        cases.append(case)
    hardware=[]
    for name, options in [('ideal',{}),('output_loss',{'transmissions':[1]*4+[.6]*2}),
                          ('late',{'schedule':FeedForward(processing_seconds=2e-9,buffer_seconds=1e-9)})]:
        h=teleportation_instrument(**options);r=h.apply(np.eye(2)/2)
        hardware.append({'case':name,**{k:v for k,v in r.items() if k.endswith('_probability')},
                         'computational_probability':r['computational_branch'].probability,
                         'flagged_completeness_residual':h.flagged_channel().completeness_residual})
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'src/plq').rglob('*.py'))}
    report={'format':'plq.scaling-hardware.v1','generated_utc':datetime.now(timezone.utc).isoformat(),
            'versions':{'plq':__version__,'numpy':np.__version__,'python':platform.python_version()},
            'source_sha256':hashes,'scaling':cases,'hardware':hardware,
            'scope':'Single-process wall times on this environment; sparse support is circuit-dependent. No full experiment reproduction.'}
    (ROOT/'benchmarks/scaling_hardware.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'scaling':cases,'hardware':hardware},indent=2))


if __name__=='__main__':
    main()
