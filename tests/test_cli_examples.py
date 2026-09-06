import json
from pathlib import Path
import runpy
import importlib.util
import pytest
from plq.cli import main, optical_scenario, memory_scenario
from plq import Precision, ResourceLimitError, FockBasis, repetition_code

ROOT=Path(__file__).resolve().parents[1]


def test_optical_json_and_cli_output(tmp_path):
    config=json.loads((ROOT/"examples/configs/optics.json").read_text())
    data=optical_scenario(config)
    assert sum(x["probability"] for x in data["probabilities"]) == pytest.approx(1)
    assert sum(x["probability"] for x in data["detector_probabilities"]) == pytest.approx(1)
    output=tmp_path/"result.json"
    main(["optics",str(ROOT/"examples/configs/optics.json"),"--output",str(output)])
    assert json.loads(output.read_text())["diagnostics"]["trace_real"] == pytest.approx(1)
    with pytest.raises(ValueError): optical_scenario({**config,"efficency":.9})


def test_memory_config_custom_code_and_cli_errors(tmp_path):
    result=memory_scenario({"code":{"file":str(ROOT/"examples/configs/custom_code.json")},"shots":50})
    assert result["failures"]==0
    with pytest.raises(SystemExit) as error: main(["hom","--overlap","2"])
    assert error.value.code==2


def test_kraus_budget_prevents_large_expansion():
    tiny=Precision(max_dimension=1024,max_kraus_bytes=100)
    with pytest.raises(ResourceLimitError): repetition_code(3,precision=tiny).recovery_channel()


@pytest.mark.parametrize("name,dependency",[("hom",None),("optical_logical",None),("custom_bosonic",None),
    ("perceval_bridge","perceval"),("pennylane_bridge","pennylane"),("surface_code_stim","pymatching"),
    ("thermal_accuracy",None),("exact_memory",None),("scalable_optics",None),("hardware_teleportation",None)])
def test_documented_example_runs(name,dependency):
    if dependency: pytest.importorskip(dependency)
    runpy.run_path(str(ROOT/"examples"/(name+".py")),run_name="__main__")
