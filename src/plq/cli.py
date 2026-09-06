"""Small reproducible command line and JSON scenario interface."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import numpy as np
from . import __version__
from .numerics import Precision, integer
from .optics import FockBasis, FockState, Circuit, independent_sources, DualRail
from .detectors import Detector, detection_probabilities, herald
from .wavepackets import wavepacket_input
from .qec import get_code, StabilizerCode, MinimumWeightDecoder, ErasureDecoder
from .simulation import MemoryNoise, simulate_memory
from .decoding import MaximumLikelihoodDecoder, exact_pauli_memory
from .experiments import source_hom
from .sources import number_distribution_from_moments


def _unknown(data, allowed):
    if not isinstance(data, dict):
        raise ValueError("Configuration sections must be JSON objects")
    extra=set(data)-set(allowed)
    if extra:
        raise ValueError(f"Unknown configuration fields: {sorted(extra)}")


def optical_scenario(config):
    """Evaluate a validated, data-only optical scenario (no imports/eval from JSON)."""
    _unknown(config,{"modes","max_photons","precision","input","steps","detectors","herald","dualrail_qubits"})
    precision=Precision(**config.get("precision",{}))
    source=config["input"]
    _unknown(source,{"occupation","sources","amplitudes"})
    cutoff=config.get("max_photons")
    if cutoff is None:
        if set(source)=={"occupation"}:
            cutoff=sum(integer(n,"occupation") for n in source["occupation"])
        elif set(source)=={"sources"}:
            cutoff=sum(max(0,len(dist)-1) for dist in source["sources"])
        elif set(source)=={"amplitudes"}:
            cutoff=max((sum(integer(n,"occupation") for n in item["occupation"])
                        for item in source["amplitudes"]),default=0)
        else:
            raise ValueError("input needs exactly one of occupation, sources, or amplitudes")
    basis=FockBasis(config["modes"],cutoff,precision=precision)
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
        if operation not in ("bs","phase","loss","thermal_loss","phase_noise","unitary","transfer"):
            raise ValueError(f"Unsupported optical operation {operation!r}")
        if operation in ("unitary", "transfer"):
            _unknown(step,{"real","imag","modes"} | ({"atol"} if operation=="transfer" else set()))
            matrix=np.array(step.pop("real"))+1j*np.array(step.pop("imag",0))
            getattr(circuit,operation)(matrix,**step)
        else:
            getattr(circuit,operation)(**step)
    result=circuit.run(state)
    payload={"format":"plq.optical-result.v1","versions":{"plq":__version__,"numpy":np.__version__,"python":platform.python_version()},
             "config":config,"model":result.model,"source_omitted_probability":omitted,
             "bath_omitted_probability":result.bath_omitted_probability,
             "transfer_residuals":result.transfer_residuals,
             "truncation_history":result.truncation_history,"numerical_trace_error":result.numerical_trace_error,
             "initial_total_cutoff":basis.max_photons,"final_total_cutoff":result.state.basis.max_photons,
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
        branch=DualRail(config["dualrail_qubits"],basis=result.state.basis).decode(result.state)
        payload["dualrail_success"]={"probability":branch.probability,"rho_real":branch.rho.real.tolist(),"rho_imag":branch.rho.imag.tolist()}
    return payload


def memory_scenario(config, *, base_directory=None):
    _unknown(config,{"code","noise","shots","rounds","seed","final_perfect_round","method","decoder","max_patterns"})
    spec=config.get("code",{"name":"repetition"})
    _unknown(spec,{"name","parameters","file"})
    if "file" in spec:
        if set(spec)!={"file"}: raise ValueError("Choose a code file OR a named factory")
        path=Path(spec["file"])
        if base_directory is not None and not path.is_absolute():
            path=Path(base_directory)/path
        code=StabilizerCode.load(path)
    else:
        code=get_code(spec.get("name","repetition"),**spec.get("parameters",{}))
    method=config.get("method","monte_carlo")
    if method not in ("monte_carlo","exact"):
        raise ValueError("method must be monte_carlo or exact")
    noise=MemoryNoise(**config.get("noise",{}))
    budget=integer(config.get("max_patterns",1000000),"max_patterns",1)
    decoder_name=config.get("decoder")
    if decoder_name is None:
        decoder=None
    elif decoder_name=="minimum_weight":
        decoder=MinimumWeightDecoder(code,max_patterns=budget)
    elif decoder_name=="erasure":
        decoder=ErasureDecoder(code,max_patterns=budget)
    elif decoder_name=="maximum_likelihood":
        if method!="exact":
            raise ValueError("The CLI maximum_likelihood decoder is a single-round model; select method=exact")
        decoder=MaximumLikelihoodDecoder(code,noise,max_patterns=budget)
    else:
        raise ValueError("decoder must be minimum_weight, erasure or maximum_likelihood")
    if method=="exact":
        unsupported=set(config)&{"shots","seed","final_perfect_round"}
        if unsupported or integer(config.get("rounds",1),"rounds",1)!=1:
            raise ValueError("Exact mode uses one recovery round; omit shots, seed and final_perfect_round, and use rounds=1")
        payload=exact_pauli_memory(code,noise,decoder=decoder,max_patterns=budget).to_dict()
    else:
        options={key:config[key] for key in ("shots","rounds","seed","final_perfect_round") if key in config}
        payload=simulate_memory(code,noise,decoder=decoder,**options).to_dict()
    return {**payload,"method":method,"config":config,"plq_version":__version__}


def environment_report():
    """Report installed capabilities without downloading optional SDKs."""
    packages={}
    for module,distribution in (("numpy","numpy"),("scipy","scipy"),("perceval","perceval-quandela"),
                               ("pennylane","pennylane"),("stim","stim"),("pymatching","pymatching"),
                               ("mpmath","mpmath"),("pytest","pytest")):
        available=importlib.util.find_spec(module) is not None
        try:
            version=importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version=None
        packages[distribution]={"available":available,"version":version}
    return {"plq":__version__,"python":platform.python_version(),"packages":packages}


def source_hom_scenario(config):
    _unknown(config, {"sources", "indistinguishability", "multiphoton_model",
                      "beamsplitter_transmission", "transmissions", "detectors", "precision"})
    distributions = []
    for source in config["sources"]:
        _unknown(source, {"probabilities", "mean_photons", "g2_zero"})
        if set(source) == {"probabilities"}:
            distributions.append(source["probabilities"])
        elif set(source) == {"mean_photons", "g2_zero"}:
            distributions.append(number_distribution_from_moments(**source))
        else:
            raise ValueError("Each source requires probabilities OR both mean_photons and g2_zero")
    options = {k: config[k] for k in ("beamsplitter_transmission", "transmissions") if k in config}
    if "detectors" in config:
        options["detectors"] = [Detector(**item) for item in config["detectors"]]
    result = source_hom(distributions, config["indistinguishability"],
                        multiphoton_model=config["multiphoton_model"],
                        precision=Precision(**config.get("precision", {})), **options)
    return {"format": "plq.source-hom.v1", "config": config, "plq_version": __version__,
            "versions": {"numpy": np.__version__, "python": platform.python_version()}, **result}


def main(argv=None):
    parser=argparse.ArgumentParser(description="PLQ photonic logical-qubit simulator")
    parser.add_argument("--version",action="version",version=__version__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor",help="Show the installed core and optional SDK versions")
    for command in ("optics","memory","source-hom"):
        child=sub.add_parser(command,help=f"Run a {command} JSON scenario")
        child.add_argument("config",type=Path)
        child.add_argument("--output",type=Path)
    hom=sub.add_parser("hom",help="Two-photon interference with overlap and propagation loss")
    overlap_group=hom.add_mutually_exclusive_group()
    overlap_group.add_argument("--overlap",type=float,help="Pure-wavepacket amplitude (default: 1)")
    overlap_group.add_argument("--indistinguishability",type=float,help="Squared overlap M in [0,1], as reported for ideal single-photon HOM")
    hom.add_argument("--transmission",type=float,default=1)
    hom.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=="doctor":
            payload=environment_report()
        elif args.command=="hom":
            from .numerics import probability
            overlap = (np.sqrt(probability(args.indistinguishability, "indistinguishability"))
                       if args.indistinguishability is not None else
                       (1. if args.overlap is None else args.overlap))
            photons=wavepacket_input(2,[0,1],[[1,overlap],[overlap,1]])
            circuit=Circuit(2).bs(0,1).loss(0,args.transmission).loss(1,args.transmission)
            p=photons.through(circuit).spatial_probabilities()
            payload={"coincidence_probability":p.get((1,1),0),"overlap_amplitude":overlap,
                     "transmission":args.transmission,"total_probability":sum(p.values()),
                     "plq_version":__version__,"model":"pure-wavepacket HOM with independent vacuum-environment loss"}
        else:
            with args.config.open(encoding="utf-8") as f: config=json.load(f)
            if args.command == "source-hom":
                payload = source_hom_scenario(config)
            else:
                payload=(optical_scenario(config) if args.command=="optics" else
                         memory_scenario(config,base_directory=args.config.resolve().parent))
        encoded=json.dumps(payload,indent=2,allow_nan=False)+"\n"
        if getattr(args,"output",None):
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(encoded,encoding="utf-8")
        else:
            print(encoded,end="")
    except (ValueError,TypeError,KeyError,OSError) as exc:
        parser.error(str(exc))


if __name__=="__main__":
    main()
