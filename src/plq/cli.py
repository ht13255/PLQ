"""Small reproducible command line and JSON scenario interface."""
import argparse
import json
from pathlib import Path
import platform
import numpy as np
from . import __version__
from .numerics import Precision
from .optics import FockBasis, FockState, Circuit, independent_sources, DualRail
from .detectors import Detector, detection_probabilities, herald
from .wavepackets import wavepacket_input
from .qec import get_code, StabilizerCode
from .simulation import MemoryNoise, simulate_memory


def _unknown(data, allowed):
    extra=set(data)-set(allowed)
    if extra:
        raise ValueError(f"Unknown configuration fields: {sorted(extra)}")


def optical_scenario(config):
    """Evaluate a validated, data-only optical scenario (no imports/eval from JSON)."""
    _unknown(config,{"modes","max_photons","precision","input","steps","detectors","herald","dualrail_qubits"})
    precision=Precision(**config.get("precision",{}))
    basis=FockBasis(config["modes"],config["max_photons"],precision=precision)
    source=config["input"]
    omitted=0.
    if set(source)=={"occupation"}:
        state=FockState.ket(basis,source["occupation"])
    elif set(source)=={"sources"}:
        result=independent_sources(basis,source["sources"])
        state,omitted=result.state,result.omitted_probability
    elif set(source)=={"amplitudes"}:
        amplitudes={}
        for item in source["amplitudes"]:
            _unknown(item,{"occupation","real","imag"})
            key=tuple(item["occupation"])
            if key in amplitudes:
                raise ValueError("Repeated input occupation")
            amplitudes[key]=complex(item.get("real",0),item.get("imag",0))
        state=FockState.amplitudes(basis,amplitudes)
    else:
        raise ValueError("input needs exactly one of occupation, sources, or amplitudes")
    circuit=Circuit(basis.modes)
    for step in config.get("steps",[]):
        step=dict(step)
        operation=step.pop("operation")
        if operation not in ("bs","phase","loss","phase_noise","unitary"):
            raise ValueError(f"Unsupported optical operation {operation!r}")
        if operation=="unitary":
            _unknown(step,{"real","imag","modes"})
            matrix=np.array(step.pop("real"))+1j*np.array(step.pop("imag",0))
            circuit.unitary(matrix,**step)
        else:
            getattr(circuit,operation)(**step)
    result=circuit.run(state)
    payload={"format":"plq.optical-result.v1","versions":{"plq":__version__,"numpy":np.__version__,"python":platform.python_version()},
             "config":config,"model":result.model,"source_omitted_probability":omitted,
             "trace_history":result.trace_history,"diagnostics":result.state.diagnostics(),
             "probabilities":[{"occupation":list(s),"probability":p} for s,p in result.probabilities().items()]}
    if "detectors" in config:
        detectors=[Detector(**item) for item in config["detectors"]]
        payload["detector_probabilities"]=[{"counts":list(c),"probability":p}
            for c,p in detection_probabilities(result.state,detectors=detectors).items()]
    if "herald" in config:
        h=dict(config["herald"])
        _unknown(h,{"modes","counts","detectors"})
        if "detectors" in h: h["detectors"]=[Detector(**d) for d in h["detectors"]]
        branch=herald(result.state,**h)
        payload["herald"]={"probability":branch.probability,"remaining_modes":branch.remaining_modes,
                           "remaining_probabilities":None if branch.remaining_state is None else
                               [{"occupation":list(s),"probability":p} for s,p in branch.remaining_state.probabilities().items()]}
    if "dualrail_qubits" in config:
        branch=DualRail(config["dualrail_qubits"],basis=basis).decode(result.state)
        payload["dualrail_success"]={"probability":branch.probability,"rho_real":branch.rho.real.tolist(),"rho_imag":branch.rho.imag.tolist()}
    return payload


def memory_scenario(config):
    _unknown(config,{"code","noise","shots","rounds","seed","final_perfect_round"})
    spec=config.get("code",{"name":"repetition"})
    _unknown(spec,{"name","parameters","file"})
    if "file" in spec:
        if set(spec)!={"file"}: raise ValueError("Choose a code file OR a named factory")
        code=StabilizerCode.load(spec["file"])
    else:
        code=get_code(spec.get("name","repetition"),**spec.get("parameters",{}))
    options={key:config[key] for key in ("shots","rounds","seed","final_perfect_round") if key in config}
    return simulate_memory(code,MemoryNoise(**config.get("noise",{})),**options).to_dict()


def main(argv=None):
    parser=argparse.ArgumentParser(description="PLQ photonic logical-qubit simulator")
    parser.add_argument("--version",action="version",version=__version__)
    sub=parser.add_subparsers(dest="command",required=True)
    for command in ("optics","memory"):
        child=sub.add_parser(command,help=f"Run a {command} JSON scenario")
        child.add_argument("config",type=Path)
        child.add_argument("--output",type=Path)
    hom=sub.add_parser("hom",help="Two-photon interference with overlap and propagation loss")
    hom.add_argument("--overlap",type=float,default=1)
    hom.add_argument("--transmission",type=float,default=1)
    hom.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=="hom":
            photons=wavepacket_input(2,[0,1],[[1,args.overlap],[args.overlap,1]])
            circuit=Circuit(2).bs(0,1).loss(0,args.transmission).loss(1,args.transmission)
            p=photons.through(circuit).spatial_probabilities()
            payload={"coincidence_probability":p.get((1,1),0),"overlap_amplitude":args.overlap,
                     "transmission":args.transmission,"total_probability":sum(p.values())}
        else:
            with args.config.open(encoding="utf-8") as f: config=json.load(f)
            payload=(optical_scenario if args.command=="optics" else memory_scenario)(config)
        encoded=json.dumps(payload,indent=2,allow_nan=False)+"\n"
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(encoded,encoding="utf-8")
        else:
            print(encoded,end="")
    except (ValueError,TypeError,KeyError,OSError) as exc:
        parser.error(str(exc))


if __name__=="__main__":
    main()
