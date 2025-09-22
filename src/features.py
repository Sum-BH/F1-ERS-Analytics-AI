# src/features.py
import pandas as pd
from pathlib import Path
from .configs import DATA_CSV, OUTPUT_DIR

def make_features(df, seg_len=120):
    df = df.copy()
    df['brake_pct'] = df['brake'].clip(0,1)*100
    df['throttle_pct'] = df['throttle'].clip(0,1)*100
    df['battery_pct'] = df['battery'].clip(0,1)*100
    gain_factor = 0.0000006
    df['time_saved'] = df['deployed'] * gain_factor * 1e6

    segments = []
    for i in range(0, len(df), seg_len):
        seg = df.iloc[i:i+seg_len]
        if seg.empty: continue
        td = float(seg['deployed'].sum())
        est = float(seg['time_saved'].sum())
        eff = est/td if td>1e-6 else 0.0
        segments.append({
            'segment_idx': i//seg_len,
            'avg_speed_kmh': float(seg['speed_kmh'].mean()),
            'total_harvest_MJ': float(seg['harvested'].sum()),
            'total_deploy_MJ': td,
            'est_time_saved_s': est,
            'efficiency_s_per_MJ': eff
        })

    lap_summary = df.groupby('lap').agg(lap_time=('time','max'), saved=('time_saved','sum')).reset_index()
    lap_summary['pred_no_deploy'] = lap_summary['lap_time'] + lap_summary['saved']
    deployed_per_lap = df.groupby('lap')['deployed'].sum().values + 1e-6
    lap_summary['efficiency_s_per_MJ'] = lap_summary['saved'] / deployed_per_lap

    return df, pd.DataFrame(segments), lap_summary

if __name__ == '__main__':
    df = pd.read_csv(DATA_CSV)
    df, segments, lap_summary = make_features(df)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'segments.csv').write_text(segments.to_csv(index=False))
    lap_summary.to_csv(out_dir / 'lap_summary.csv', index=False)
    print('Features created and saved to outputs/')
