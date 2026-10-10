import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1]); out = sys.argv[2]
def stab(df, px, py):
    x, y = df[px].values, df[py].values
    w = df.h20 * x + df.h21 * y + df.h22
    return (df.h00 * x + df.h01 * y + df.h02) / w, (df.h10 * x + df.h11 * y + df.h12) / w
wy = (df.l_wri_y + df.r_wri_y) / 2; wx = (df.l_wri_x + df.r_wri_x) / 2
df["wx"], df["wy"] = wx, wy
sx, sy = stab(df, "wx", "wy")
fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
ax[0].plot(df.t, wy, label="wrist y raw (scene px)"); ax[0].plot(df.t, sy, label="wrist y stabilised"); ax[0].invert_yaxis(); ax[0].legend()
ax[1].plot(df.t, df.l_wri_c, label="l_wri conf"); ax[1].plot(df.t, df.r_wri_c, label="r_wri conf"); ax[1].plot(df.t, df.p_conf, label="person conf"); ax[1].legend()
ax[2].plot(df.t, df.ref_id, label="ref_id"); ax[2].plot(df.t, df.seg_id, label="seg_id"); ax[2].legend()
ax[3].plot(df.t, df.h_inliers, label="inliers"); ax[3].set_yscale("log"); ax[3].legend()
plt.tight_layout(); plt.savefig(out, dpi=70)
print(df[["p_conf","l_wri_c","r_wri_c","h_inliers"]].describe().round(2))
