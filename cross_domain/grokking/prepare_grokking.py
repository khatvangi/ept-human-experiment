"""
prepare_grokking.py — load grokking training curves and run EPT inference

two data sources:
  1. /storage/EPT/grokking_timeseries.csv  (1200 rows, epoch step=10)
  2. /storage/EPT/ept_manuscript/data_grok_cos1_seed0.json  (epoch step=500)

applies the domain-agnostic inference module to test_accuracy curves
and saves results to grokking_analysis.json.
"""

import sys
import json
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, '/storage/EPT/ept_human_experiment')
from inference import detect_transitions

OUTPUT_PATH = '/storage/EPT/ept_human_experiment/cross_domain/grokking/grokking_analysis.json'


def load_csv_source():
    """load the dense timeseries CSV (1200 epochs, step=10)."""
    df = pd.read_csv('/storage/EPT/grokking_timeseries.csv')
    print(f"[csv] loaded {len(df)} rows, epochs {df.epoch.min()}-{df.epoch.max()}")
    return df


def load_json_sources():
    """load all seed JSON files from ept_manuscript/."""
    pattern = '/storage/EPT/ept_manuscript/data_grok_*seed*.json'
    files = sorted(glob.glob(pattern))
    print(f"[json] found {len(files)} seed file(s): {[f.split('/')[-1] for f in files]}")

    datasets = {}
    for fp in files:
        name = fp.split('/')[-1].replace('.json', '')
        with open(fp) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"  {name}: {len(df)} rows, epochs {df.epoch.min()}-{df.epoch.max()}")
        datasets[name] = df
    return datasets


def run_inference(name, test_acc, chance_level=0.0):
    """run the transition detection pipeline on a test_acc series."""
    print(f"\n{'='*60}")
    print(f"inference on: {name}  ({len(test_acc)} timepoints)")
    print(f"{'='*60}")

    result = detect_transitions(
        accuracy_series=test_acc,
        chance_level=chance_level,
        min_improvement=0.15,
    )

    # print key results
    print(f"  classification : {result['classification']['label']}")
    print(f"  summary        : {result['summary']}")
    psi_stats = result['psi']
    if psi_stats:
        print(f"  Ψ peak value   : {psi_stats['peak_value']:.4f}")
        print(f"  Ψ peak index   : {psi_stats['peak_idx']}")
        print(f"  Ψ z-score      : {psi_stats.get('z_score', 'N/A')}")
        print(f"  Ψ spike ratio  : {psi_stats.get('spike_ratio', 'N/A')}")
        print(f"  Ψ localization : {psi_stats.get('localization', 'N/A')}")
    print(f"  changepoints   : {result['changepoints']}")
    mc = result['model_comparison']
    print(f"  best model     : {mc['best_model']} (ΔBIC={mc['delta_bic']:.1f})")

    return result


def main():
    # -- load data --
    csv_df = load_csv_source()
    json_datasets = load_json_sources()

    # -- run inference on each source --
    all_results = {}

    # 1. dense CSV — downsample to ~200 points for tractable HMM fitting
    #    (original is 1200 rows at step=10; take every 6th → ~200 points)
    step = max(1, len(csv_df) // 200)
    csv_sub = csv_df.iloc[::step].reset_index(drop=True)
    print(f"[csv] downsampled {len(csv_df)} → {len(csv_sub)} rows (step={step})")
    res_csv = run_inference(
        "csv_dense",
        csv_sub['test_acc'].values,
        chance_level=0.0,
    )
    all_results['csv_dense'] = {
        'source': 'grokking_timeseries.csv',
        'n_epochs': int(csv_df.epoch.max()),
        'n_timepoints_original': len(csv_df),
        'n_timepoints_analyzed': len(csv_sub),
        'classification': res_csv['classification'],
        'psi': res_csv['psi'],
        'changepoints': res_csv['changepoints'],
        'model_comparison': {k: v for k, v in res_csv['model_comparison'].items()
                            if k != 'models'},  # skip bulky model objects
        'summary': res_csv['summary'],
    }

    # 2. each JSON seed file (coarser, ~36 points)
    for name, df in json_datasets.items():
        res = run_inference(
            name,
            df['test_acc'].values,
            chance_level=0.0,
        )
        all_results[name] = {
            'source': f'{name}.json',
            'n_epochs': int(df.epoch.max()),
            'n_timepoints': len(df),
            'classification': res['classification'],
            'psi': res['psi'],
            'changepoints': res['changepoints'],
            'model_comparison': {k: v for k, v in res['model_comparison'].items()
                                if k != 'models'},
            'summary': res['summary'],
        }

    # -- save --
    # convert any numpy types for json serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"not serializable: {type(obj)}")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\n\nsaved analysis to {OUTPUT_PATH}")
    print(f"datasets analyzed: {list(all_results.keys())}")


if __name__ == '__main__':
    main()
