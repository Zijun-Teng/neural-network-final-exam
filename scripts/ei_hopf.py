from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ei"
OUT.mkdir(parents=True, exist_ok=True)


def relu(x):
    return np.maximum(x, 0.0)


def simulate(tau_i, dt=0.002, tmax=80.0):
    tau_e = 1.0
    mee, mei, mie, mii = 2.0, 4.0, 1.0, 1.0
    he, hi = 0.5, 0.0
    n = int(tmax / dt)
    v = np.array([0.54, 0.22], dtype=float)
    trace = []
    for k in range(n):
        ve, vi = v
        dve = (-ve + relu(mee * ve - mei * vi + he)) / tau_e
        dvi = (-vi + relu(mie * ve - mii * vi + hi)) / tau_i
        v += dt * np.array([dve, dvi])
        if k % 20 == 0:
            trace.append(v.copy())
    return np.array(trace)


def eigs(tau_i):
    tau_e = 1.0
    mee, mei, mie, mii = 2.0, 4.0, 1.0, 1.0
    j = np.array([[(mee - 1) / tau_e, -mei / tau_e], [mie / tau_i, -(mii + 1) / tau_i]])
    return np.linalg.eigvals(j)


def positive_branch_equilibrium(h_e):
    # For the parameter values used below and h_I=0, the active-active
    # equilibrium is (v_E, v_I)=(h_E, h_E/2). It is admissible only for h_E>0.
    return h_e, 0.5 * h_e


def draw_phase(curves):
    w, h = 760, 520
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    margin = 55
    draw.rectangle((margin, margin, w - margin, h - margin), outline=(30, 30, 30))
    allp = np.vstack(list(curves.values()))
    xmin, ymin = allp.min(axis=0) - 0.02
    xmax, ymax = allp.max(axis=0) + 0.02

    def map_pt(p):
        x = margin + (p[0] - xmin) / (xmax - xmin) * (w - 2 * margin)
        y = h - margin - (p[1] - ymin) / (ymax - ymin) * (h - 2 * margin)
        return int(x), int(y)

    colors = {"tauI=1.5": (31, 119, 180), "tauI=2.0": (44, 160, 44), "tauI=2.5": (214, 39, 40)}
    ytxt = 18
    for label, pts in curves.items():
        mapped = [map_pt(p) for p in pts]
        draw.line(mapped, fill=colors[label], width=2)
        draw.text((60, ytxt), label, fill=colors[label])
        ytxt += 20
    draw.text((w // 2 - 35, h - 35), "v_E", fill=(0, 0, 0))
    draw.text((12, h // 2), "v_I", fill=(0, 0, 0))
    img.save(OUT / "ei_phase.png")


def draw_bifurcation():
    w, h = 760, 440
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    margin = 55
    draw.rectangle((margin, 30, w - margin, h - margin), outline=(30, 30, 30))
    taus = np.linspace(1.05, 3.2, 240)
    real_parts = np.array([np.max(eigs(t).real) for t in taus])
    ymin, ymax = -0.55, 0.35

    def map_pt(t, r):
        x = margin + (t - taus.min()) / (taus.max() - taus.min()) * (w - 2 * margin)
        y = h - margin - (r - ymin) / (ymax - ymin) * (h - margin - 30)
        return int(x), int(y)

    zero_y = map_pt(taus[0], 0.0)[1]
    draw.line((margin, zero_y, w - margin, zero_y), fill=(170, 170, 170), width=1)
    pts = [map_pt(t, r) for t, r in zip(taus, real_parts)]
    draw.line(pts, fill=(214, 39, 40), width=3)
    crit = map_pt(2.0, 0.0)
    draw.ellipse((crit[0] - 4, crit[1] - 4, crit[0] + 4, crit[1] + 4), fill=(0, 0, 0))
    draw.text((crit[0] + 8, crit[1] - 18), "tau_I*=2", fill=(0, 0, 0))
    draw.text((w // 2 - 45, h - 35), "tau_I", fill=(0, 0, 0))
    draw.text((10, 38), "max Re(lambda)", fill=(0, 0, 0))
    img.save(OUT / "ei_bifurcation.png")


def main():
    curves = {f"tauI={tau:.1f}": simulate(tau) for tau in (1.5, 2.0, 2.5)}
    draw_phase(curves)
    draw_bifurcation()
    lines = [
        "E-I Hopf verification",
        "Parameters: tauE=1, MEE=2, MEI=4, MIE=1, MII=1, hE=0.5, hI=0",
        "Critical tauI: 2",
        "Active-active branch for hE>0: vE*=hE, vI*=hE/2; Jacobian is independent of hE.",
        "tauI,lambda1,lambda2,final_vE,final_vI",
    ]
    for tau in (1.5, 2.0, 2.5):
        ev = eigs(tau)
        final = curves[f"tauI={tau:.1f}"][-1]
        lines.append(
            f"{tau:.1f},{ev[0].real:.6f}{ev[0].imag:+.6f}i,"
            f"{ev[1].real:.6f}{ev[1].imag:+.6f}i,{final[0]:.6f},{final[1]:.6f}"
        )
    lines.append("hE,branch_vE,branch_vI,admissible")
    for h_e in (-0.2, 0.0, 0.5, 1.0):
        ve, vi = positive_branch_equilibrium(h_e)
        lines.append(f"{h_e:.1f},{ve:.6f},{vi:.6f},{h_e > 0}")
    (OUT / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
