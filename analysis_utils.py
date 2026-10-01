"""Measured assignment outputs and explicit missing-baseline handling."""
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd


def percent_change(baseline, scenario):
    baseline = pd.Series(baseline, dtype=float)
    scenario = pd.Series(scenario, index=baseline.index, dtype=float)
    valid = np.isfinite(baseline) & np.isfinite(scenario) & (baseline > 0)
    change = pd.Series(np.nan, index=baseline.index)
    change.loc[valid] = 100 * (scenario[valid] - baseline[valid]) / baseline[valid]
    status = pd.Series('missing', index=baseline.index)
    status.loc[valid] = 'measured'
    zero = (baseline == 0) & np.isfinite(scenario)
    status.loc[zero] = 'zero_baseline'
    status.loc[zero & (scenario > 0)] = 'new_volume'
    return change, status


def assignment_metrics(raw):
    observed = raw['ODME_obs_count'].to_numpy(float)
    valid = np.isfinite(observed) & (observed > 0)
    out = dict(total_count=len(raw), observation_count=int(valid.sum()),
               excluded_nonpositive_or_missing=int((~valid).sum()), evaluation='calibration_fit')
    for stage in ['before', 'after']:
        predicted = raw['ODME_volume_' + stage].to_numpy(float)
        if np.any(valid & ~np.isfinite(predicted)):
            raise ValueError('Nonfinite prediction at an observed sensor')
        errors = abs(predicted[valid] - observed[valid])
        out['MAE_' + stage] = float(errors.mean()) if errors.size else np.nan
        out['MAPE_' + stage] = float((errors / observed[valid]).mean() * 100) if errors.size else np.nan
    return out


def hourly_assignment(raw, fractions, column):
    fractions = pd.Series(fractions).reindex(range(24))
    if not np.isfinite(fractions).all() or (fractions < 0).any() or not np.isclose(fractions.sum(), 1):
        raise ValueError('A complete, normalized 24-hour profile is required')
    daily = raw.groupby(['from_node_id', 'to_node_id'])[column].sum(min_count=1).reset_index()
    if not np.isfinite(daily[column]).all():
        raise ValueError('Missing assignment volumes; rerun the simulation')
    frames = []
    for hour in range(24):
        frame = daily[['from_node_id', 'to_node_id']].copy()
        frame['volume_hourly'] = daily[column] * fractions.loc[hour]
        frame['hour'] = hour
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.attrs['source_column'] = column
    return result


def save_assignment(frame, path, source_paths, scenario, profile_date):
    path = Path(path)
    frame.to_pickle(path)
    metadata = dict(source_column=frame.attrs['source_column'], scenario=scenario,
                    profile_date=str(profile_date), output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    inputs={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths})
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def bridge_emission_comparison(emissions, road, bridge_csv):
    bridge_csv = Path(bridge_csv)
    if not bridge_csv.exists():
        warnings.warn('Bridge comparison skipped: provide a verified bridge_links.csv with linkID and source columns')
        return pd.DataFrame(), dict(status='missing_bridge_catalog', catalog=str(bridge_csv))
    catalog = pd.read_csv(bridge_csv, dtype=str)
    if not {'linkID', 'source'}.issubset(catalog.columns) or catalog.empty or catalog['source'].fillna('').str.strip().eq('').any():
        raise ValueError('Bridge catalog needs verified linkID values and their sources')
    ids = set(catalog.linkID)
    if not ids.issubset(set(road.linkID)):
        raise ValueError('Bridge catalog contains links absent from the road network')
    bridge = emissions.linkID.isin(ids)
    if not bridge.any() or bridge.all():
        return pd.DataFrame(), dict(status='insufficient_bridge_or_reference_coverage')
    ref = emissions[~bridge].groupby(['pollutantID', 'Hour']).emrate_.mean().rename('reference')
    br = emissions[bridge].groupby(['pollutantID', 'Hour']).emrate_.mean().rename('bridge')
    result = pd.concat([ref, br], axis=1).reset_index()
    result['diff_pct'], result['status'] = percent_change(result.reference, result.bridge)
    return result, dict(status='measured', bridge_links=len(ids), catalog_sha256=hashlib.sha256(bridge_csv.read_bytes()).hexdigest())


def read_assignment(path, stage):
    path = Path(path)
    metadata_path = path.with_suffix('.json')
    if stage not in {'before', 'after'} or not metadata_path.exists():
        raise ValueError('Legacy assignment lacks stage provenance; rerun 2.2_dtalite_analysis.py')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('source_column') != 'ODME_volume_' + stage or metadata.get('output_sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Assignment stage or checksum mismatch; regenerate paired outputs')
    frame = pd.read_pickle(path)
    if frame.attrs.get('source_column') != metadata['source_column']:
        raise ValueError('Assignment source column mismatch')
    return frame
