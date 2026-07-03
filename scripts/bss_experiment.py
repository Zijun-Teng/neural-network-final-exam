from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "BSS"
if not DATA.exists():
    DATA = ROOT / "附件" / "BSS"
OUT = ROOT / "outputs" / "bss"
OUT.mkdir(parents=True, exist_ok=True)


def load_mixtures():
    files = sorted(DATA.glob("*.wav"))
    rates = []
    signals = []
    for f in files:
        fs, x = wavfile.read(f)
        rates.append(fs)
        signals.append(x.astype(np.float64))
    if len(set(rates)) != 1:
        raise RuntimeError("Sampling rates differ.")
    x = np.vstack(signals)
    x = x - x.mean(axis=1, keepdims=True)
    x = x / (x.std(axis=1, keepdims=True) + 1e-12)
    return rates[0], x, files


def whiten(x):
    cov = x @ x.T / x.shape[1]
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    k = np.diag(1.0 / np.sqrt(evals + 1e-12)) @ evecs.T
    z = k @ x
    return z, k


def symmetric_decorrelation(w):
    d, e = np.linalg.eigh(w @ w.T)
    return (e @ np.diag(1.0 / np.sqrt(d + 1e-12)) @ e.T) @ w


def fastica(z, max_iter=1000, tol=1e-7, seed=7):
    rng = np.random.default_rng(seed)
    n, t = z.shape
    w = symmetric_decorrelation(rng.normal(size=(n, n)))
    for _ in range(max_iter):
        y = w @ z
        g = np.tanh(y)
        gp = 1.0 - g**2
        w_new = (g @ z.T) / t - np.diag(gp.mean(axis=1)) @ w
        w_new = symmetric_decorrelation(w_new)
        lim = np.max(np.abs(np.abs(np.diag(w_new @ w.T)) - 1.0))
        w = w_new
        if lim < tol:
            break
    return w @ z


def amuse(z, lag=1):
    c_tau = z[:, lag:] @ z[:, :-lag].T / (z.shape[1] - lag)
    c_tau = 0.5 * (c_tau + c_tau.T)
    evals, evecs = np.linalg.eigh(c_tau)
    order = np.argsort(np.abs(evals))[::-1]
    return evecs[:, order].T @ z


def quality(y):
    y = y - y.mean(axis=1, keepdims=True)
    y = y / (y.std(axis=1, keepdims=True) + 1e-12)
    corr = np.corrcoef(y)
    off = corr - np.eye(corr.shape[0])
    mean_abs_corr = float(np.sum(np.abs(off)) / (corr.size - corr.shape[0]))
    lag = 1
    lag_cov = y[:, lag:] @ y[:, :-lag].T / (y.shape[1] - lag)
    lag_off = lag_cov - np.diag(np.diag(lag_cov))
    mean_abs_lag_corr = float(np.sum(np.abs(lag_off)) / (lag_cov.size - lag_cov.shape[0]))
    kurt = np.mean(y**4, axis=1) - 3.0
    return mean_abs_corr, mean_abs_lag_corr, kurt


def write_wavs(fs, name, y):
    y = y - y.mean(axis=1, keepdims=True)
    y = y / (np.max(np.abs(y), axis=1, keepdims=True) + 1e-12)
    for i, s in enumerate(y, 1):
        wavfile.write(OUT / f"{name}_source_{i}.wav", fs, (0.95 * s * 32767).astype(np.int16))


def waveform_image(series, labels, path):
    w, h = 1100, max(560, 80 * len(series) + 70)
    margin_l, margin_r = 70, 20
    row_h = (h - 40) // len(series)
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]
    for idx, (sig, label) in enumerate(zip(series, labels)):
        top = 20 + idx * row_h
        mid = top + row_h // 2
        draw.text((10, top + 6), label, fill=(20, 20, 20))
        draw.line((margin_l, mid, w - margin_r, mid), fill=(210, 210, 210))
        step = max(1, len(sig) // (w - margin_l - margin_r))
        s = sig[::step][: w - margin_l - margin_r]
        s = s / (np.max(np.abs(s)) + 1e-12)
        pts = [(margin_l + i, int(mid - v * (row_h * 0.36))) for i, v in enumerate(s)]
        draw.line(pts, fill=colors[idx % len(colors)], width=1)
    img.save(path)


def main():
    fs, x, files = load_mixtures()
    z, _ = whiten(x)
    pca_sources = z
    amuse_sources = amuse(z, lag=1)
    ica_sources = fastica(z)

    write_wavs(fs, "pca", pca_sources)
    write_wavs(fs, "amuse", amuse_sources)
    write_wavs(fs, "fastica", ica_sources)
    waveform_image(
        [
            x[0],
            x[1],
            x[2],
            amuse_sources[0],
            amuse_sources[1],
            amuse_sources[2],
            ica_sources[0],
            ica_sources[1],
            ica_sources[2],
        ],
        [
            "mix1",
            "mix2",
            "mix3",
            "AMUSE s1",
            "AMUSE s2",
            "AMUSE s3",
            "FastICA s1",
            "FastICA s2",
            "FastICA s3",
        ],
        OUT / "waveforms.png",
    )

    mix_corr, mix_lag, mix_kurt = quality(x)
    pca_corr, pca_lag, pca_kurt = quality(pca_sources)
    amuse_corr, amuse_lag, amuse_kurt = quality(amuse_sources)
    ica_corr, ica_lag, ica_kurt = quality(ica_sources)

    lines = [
        "Blind source separation experiment",
        f"Sampling rate: {fs} Hz",
        f"Samples per channel: {x.shape[1]}",
        f"Files: {', '.join(f.name for f in files)}",
        "",
        "method,mean_abs_offdiag_corr,mean_abs_lag1_offdiag_corr,kurtosis_1,kurtosis_2,kurtosis_3,mean_abs_kurtosis",
        f"mixtures,{mix_corr:.6f},{mix_lag:.6f},{mix_kurt[0]:.6f},{mix_kurt[1]:.6f},{mix_kurt[2]:.6f},{np.mean(np.abs(mix_kurt)):.6f}",
        f"PCA whitening,{pca_corr:.6f},{pca_lag:.6f},{pca_kurt[0]:.6f},{pca_kurt[1]:.6f},{pca_kurt[2]:.6f},{np.mean(np.abs(pca_kurt)):.6f}",
        f"AMUSE,{amuse_corr:.6f},{amuse_lag:.6f},{amuse_kurt[0]:.6f},{amuse_kurt[1]:.6f},{amuse_kurt[2]:.6f},{np.mean(np.abs(amuse_kurt)):.6f}",
        f"FastICA,{ica_corr:.6f},{ica_lag:.6f},{ica_kurt[0]:.6f},{ica_kurt[1]:.6f},{ica_kurt[2]:.6f},{np.mean(np.abs(ica_kurt)):.6f}",
    ]
    (OUT / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
