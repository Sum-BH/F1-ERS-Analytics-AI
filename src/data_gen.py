# src/data_gen.py
import numpy as np, pandas as pd, math
from pathlib import Path
from datetime import datetime
from .configs import DATA_CSV

def generate_synthetic_monza(n_laps=6, points_per_lap=1200, seed=42):
    rnd = np.random.RandomState(seed)
    rows = []
    track_length = 5793.0
    braking_zones = [(300,450),(1150,1300),(2250,2450),(3350,3550),(4600,4800)]
    avg_dt = 0.08

    for lap in range(n_laps):
        battery = 0.75 + rnd.rand()*0.2
        time_acc = 0.0
        for i in range(points_per_lap):
            s = (i/points_per_lap)*track_length
            base_speed = 200 + 140 * math.sin(2*math.pi*(s/track_length))
            speed_kmh = np.clip(base_speed + rnd.normal(0,6), 80, 340)
            speed = speed_kmh / 3.6
            in_brake = any(a <= s <= b for a,b in braking_zones)
            brake = float(rnd.rand() * (0.9 if in_brake else 0.04))
            throttle = float(rnd.rand() * (0.95 if not in_brake else 0.2))

            harvested = deployed = 0.0
            if brake > 0.6 and battery < 0.99:
                harvested = 0.01 * brake * (1-battery)
                battery = min(1.0, battery + harvested)
            if (not in_brake) and throttle > 0.6 and battery > 0.12 and rnd.rand() > 0.78:
                deployed = 0.02 * (0.2 + 0.3*throttle)
                battery = max(0.0, battery - deployed)

            dt = avg_dt * (0.95 if deployed>0 else 1.0)
            time_acc += dt

            rows.append({
                "lap":lap,"idx":i,"s":s,"speed":speed,"speed_kmh":speed_kmh,
                "throttle":throttle,"brake":brake,"gear":int(np.clip(speed_kmh/40,1,8)),
                "battery":battery,"harvested":harvested,"deployed":deployed,"time":time_acc
            })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_synthetic_monza()
    Path(DATA_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_CSV, index=False)
    print(f"Generated telemetry: laps={df['lap'].nunique()}, rows={len(df)}")
