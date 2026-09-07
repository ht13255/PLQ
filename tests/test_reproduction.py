import copy
import json
import pytest
from plq.reproduction import compare_experiment, content_hash, evidence_record
from plq.cli import source_hom_scenario, main


def fixture(tmp_path, origin='synthetic'):
    data = {'origin': origin, 'shots': 100, 'bins': [{'outcome': [1], 'count': 70}], 'other_count': 30}
    (tmp_path/'counts.json').write_text(json.dumps(data))
    scenario = {'modes': 1, 'input': {'occupation': [1]},
                'steps': [{'operation': 'loss', 'mode': 0, 'transmission': .7}]}
    (tmp_path/'scenario.json').write_text(json.dumps(scenario))
    return {'dataset': 'counts.json', 'dataset_sha256': content_hash(tmp_path/'counts.json'),
            'scenario': 'scenario.json', 'observable': 'photon_occupation',
            'protocol': {'trial_definition': 'One pulse, including no detection', 'selection': 'all_trials',
                         'independent_trials': True, 'same_observable': True, 'fixed_parameters': True},
            'calibration': {'transmission': 'Synthetic .7 fixture; no experimental calibration'},
            'sources': ['Synthetic unit-test fixture, not measured data']}


@pytest.mark.parametrize('origin,category', [('synthetic','synthetic_validation'),('experimental','experimental_data_comparison')])
def test_comparison_does_not_promote_an_origin_declaration(tmp_path, origin, category):
    manifest = fixture(tmp_path, origin)
    report = compare_experiment(manifest, base_directory=tmp_path, bootstrap_samples=100)
    assert report['evidence']['category'] == category
    assert report['evidence']['experimental_reproduction'] is False
    assert report['bootstrap_p_value'] == 1
    assert report['bins'][-1]['expected'] == pytest.approx(30)
    with pytest.raises(ValueError): evidence_record('experimental_reproduction')


def test_hash_denominator_protocol_and_unknown_fields(tmp_path):
    manifest = fixture(tmp_path)
    for key, value in [('dataset_sha256', 'wrong'), ('made_up', True)]:
        with pytest.raises(ValueError): compare_experiment({**manifest, key: value}, base_directory=tmp_path)
    for key, value in [('selection','accepted_only'), ('fixed_parameters',False), ('same_observable',False)]:
        changed=copy.deepcopy(manifest);changed['protocol'][key]=value
        with pytest.raises(ValueError): compare_experiment(changed, base_directory=tmp_path)
    data = json.loads((tmp_path/'counts.json').read_text()); data['other_count'] = 0
    (tmp_path/'counts.json').write_text(json.dumps(data))
    manifest['dataset_sha256'] = content_hash(tmp_path/'counts.json')
    with pytest.raises(ValueError, match='All bin counts'): compare_experiment(manifest, base_directory=tmp_path)


def test_impossible_observation_is_json_safe(tmp_path):
    manifest = fixture(tmp_path)
    scenario=json.loads((tmp_path/'scenario.json').read_text());scenario['steps'][0]['transmission']=0
    (tmp_path/'scenario.json').write_text(json.dumps(scenario))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    main(['compare-experiment', str(tmp_path/'manifest.json'), '--output', str(tmp_path/'result.json')])
    report=json.loads((tmp_path/'result.json').read_text())
    assert report['impossible_observed_event'] is True
    assert report['bootstrap_p_value'] == 0
    assert report['deviance'] is None


def test_paper_run_is_parameter_reproduction():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'examples/papers/somaschi_2016.json').read_text())
    r=source_hom_scenario(config)
    assert r['evidence']['category']=='paper_parameter_reproduction'
    assert not r['evidence']['experimental_reproduction']
    assert r['evidence']['sources']


def test_rare_nonzero_expectation_avoids_likelihood_ratio_overflow(tmp_path):
    manifest = fixture(tmp_path)
    scenario = json.loads((tmp_path/'scenario.json').read_text())
    scenario['steps'][0]['transmission'] = 1e-310
    (tmp_path/'scenario.json').write_text(json.dumps(scenario))
    report = compare_experiment(manifest, base_directory=tmp_path, bootstrap_samples=20)
    assert not report['impossible_observed_event']
    assert report['deviance'] > 1000
    json.dumps(report, allow_nan=False)
