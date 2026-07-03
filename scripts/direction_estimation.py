from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "direction"
OUT.mkdir(parents=True, exist_ok=True)


def tuning(s, prefs, sigma):
    return np.exp(-0.5 * ((s - prefs) / sigma) ** 2)


def fisher_information(s, prefs, sigma, t_window):
    f = tuning(s, prefs, sigma)
    return t_window * np.sum(f * ((s - prefs) ** 2) / sigma**4)


def mle_grid(counts, grid, prefs, sigma, t_window):
    rates = t_window * np.exp(-0.5 * ((grid[:, None] - prefs[None, :]) / sigma) ** 2)
    rates = np.maximum(rates, 1e-14)
    ll = counts @ np.log(rates).T - rates.sum(axis=1)
    return grid[int(np.argmax(ll))]


def draw_tuning(prefs, sigma, path):
    w, h = 900, 460
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    margin = 58
    draw.rectangle((margin, 30, w - 24, h - margin), outline=(30, 30, 30))
    grid = np.linspace(0.0, 180.0, 500)
    shown = np.linspace(0, len(prefs) - 1, 9).astype(int)
    palette = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14)]
    for k, idx in enumerate(shown):
        y = tuning(grid, prefs[idx], sigma)
        pts = []
        for gx, gy in zip(grid, y):
            x = margin + gx / 180.0 * (w - margin - 24)
            yy = h - margin - gy * (h - margin - 30)
            pts.append((int(x), int(yy)))
        draw.line(pts, fill=palette[k % len(palette)], width=2)
    draw.text((w // 2 - 60, h - 35), "stimulus direction (degree)", fill=(0, 0, 0))
    draw.text((10, 40), "f_a(s)", fill=(0, 0, 0))
    img.save(path)


def draw_hist(estimates, true_s, crlb, path):
    w, h = 900, 460
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    margin = 58
    draw.rectangle((margin, 30, w - 24, h - margin), outline=(30, 30, 30))
    bins = np.linspace(true_s - 18, true_s + 18, 37)
    hist, edges = np.histogram(estimates, bins=bins)
    maxh = max(1, int(hist.max()))
    for c, left, right in zip(hist, edges[:-1], edges[1:]):
        x0 = margin + (left - bins[0]) / (bins[-1] - bins[0]) * (w - margin - 24)
        x1 = margin + (right - bins[0]) / (bins[-1] - bins[0]) * (w - margin - 24)
        y0 = h - margin
        y1 = h - margin - c / maxh * (h - margin - 30)
        draw.rectangle((int(x0), int(y1), int(x1), int(y0)), fill=(31, 119, 180), outline="white")
    x_true = margin + (true_s - bins[0]) / (bins[-1] - bins[0]) * (w - margin - 24)
    draw.line((int(x_true), 30, int(x_true), h - margin), fill=(214, 39, 40), width=3)
    draw.text((margin + 10, 42), f"true s={true_s:.1f}, CRLB={crlb:.3f}", fill=(0, 0, 0))
    draw.text((w // 2 - 70, h - 35), "estimated direction (degree)", fill=(0, 0, 0))
    draw.text((10, 40), "count", fill=(0, 0, 0))
    img.save(path)


def main():
    rng = np.random.default_rng(20260703)
    n_neurons = 61
    prefs = np.linspace(0.0, 180.0, n_neurons)
    sigma = 18.0
    t_window = 25.0
    true_s = 72.0
    trials = 2000
    grid = np.linspace(0.0, 180.0, 1801)

    lam = t_window * tuning(true_s, prefs, sigma)
    estimates = np.empty(trials)
    for i in range(trials):
        counts = rng.poisson(lam)
        estimates[i] = mle_grid(counts, grid, prefs, sigma, t_window)

    bias = float(estimates.mean() - true_s)
    variance = float(estimates.var(ddof=1))
    mse = float(np.mean((estimates - true_s) ** 2))
    info = float(fisher_information(true_s, prefs, sigma, t_window))
    crlb = 1.0 / info

    draw_tuning(prefs, sigma, OUT / "tuning_curves.png")
    draw_hist(estimates, true_s, crlb, OUT / "mle_histogram.png")

    lines = [
        "Poisson population direction estimation",
        f"neurons: {n_neurons}",
        f"preferred directions: equally spaced in [0, 180]",
        f"sigma: {sigma}",
        f"observation window: {t_window}",
        f"true direction: {true_s}",
        f"Monte Carlo trials: {trials}",
        "bias,variance,mse,fisher_information,crlb",
        f"{bias:.6f},{variance:.6f},{mse:.6f},{info:.6f},{crlb:.6f}",
    ]
    (OUT / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
