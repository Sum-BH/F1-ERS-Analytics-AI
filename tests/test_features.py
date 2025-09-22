import pandas as pd
from src.features import make_features

def test_make_features_basic():
    # small synthetic table with 2 laps and 10 points each
    rows = []
    for lap in range(2):
        for i in range(10):
            rows.append({
                'lap': lap, 'idx': i, 's': i*10.0, 'speed': 50.0, 'speed_kmh': 180.0,
                'throttle': 0.1, 'brake': 0.0, 'gear': 3,
                'battery': 0.8, 'harvested': 0.0, 'deployed': 0.0, 'time': i*0.1
            })
    df = pd.DataFrame(rows)
    df_out, segments, lap_summary = make_features(df, seg_len=5)
    assert 'lap_time' in lap_summary.columns
    assert segments.shape[0] > 0
