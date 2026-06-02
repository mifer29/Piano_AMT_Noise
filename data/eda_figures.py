"""
MAESTRO Dataset: Thesis EDA Figures

Generates figures for the LaTeX thesis (Times New Roman,
booktabs style, PDF vector output).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import pretty_midi

warnings.filterwarnings("ignore")

# CONFIG 
MAESTRO_DIR    = "/export/clusterdata/mflara/maestro"   
CSV_NAME       = "maestro-v3.0.0.csv"
OUTPUT_DIR     = "eda_figures"
MAX_MIDI_FILES = 200   # set None to analyse all files

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── MATPLOTLIB GLOBAL STYLE  (LaTeX + Times New Roman) ───────────────────────
mpl.rcParams.update({
    # Use LaTeX rendering so all text matches your thesis font
    "text.usetex":          True,
    "text.latex.preamble":  r"\usepackage{times}\usepackage{amsmath}",
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],

    # Font sizes (match thesis body ~10-11pt)
    "font.size":            10,
    "axes.titlesize":       11,
    "axes.labelsize":       10,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,

    # Clean axes — no top/right spines
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.8,
    "axes.grid":            False,

    # Ticks
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,

    # Figure
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,

    # Lines & patches
    "lines.linewidth":      1.2,
    "patch.linewidth":      0.5,
})

# Colour palette — muted, greyscale-friendly, print-safe
SPLIT_COLORS = {
    "train":      "#2166AC",
    "validation": "#F4A582",
    "test":       "#4DAC26",
}
MAIN_COLOR = "#2166AC"
ACCENT     = "#D6604D"

# ── HELPERS ───────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [ok] {name}")

# ── 1. LOAD METADATA ──────────────────────────────────────────────────────────
csv_path = os.path.join(MAESTRO_DIR, CSV_NAME)
df = pd.read_csv(csv_path)
df["split"] = df["split"].str.strip().str.lower()

print("=" * 60)
print("MAESTRO EDA")
print("=" * 60)

# ── 2. FIGURE 1 — Duration distribution ──────────────────────────────────────
# figsize width ≈ \textwidth for a standard A4 single-column thesis
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6))
dur_min = df["duration"] / 60

ax = axes[0]
ax.hist(dur_min, bins=40, color=MAIN_COLOR, edgecolor="white",
        linewidth=0.4, alpha=0.85)
med = dur_min.median()
ax.axvline(med, color=ACCENT, linestyle="--", linewidth=1.0,
           label=r"Median: {:.1f} min".format(med))
ax.set_xlabel("Duration (min)")
ax.set_ylabel("Recordings")
ax.set_title("Full dataset")
ax.legend(frameon=False)

ax = axes[1]
for split, color in SPLIT_COLORS.items():
    grp = df[df["split"] == split]["duration"] / 60
    ax.hist(grp, bins=30, color=color, edgecolor="white",
            linewidth=0.4, alpha=0.75, label=split.capitalize())
ax.set_xlabel("Duration (min)")
ax.set_title("By split")
ax.legend(frameon=False)

fig.tight_layout(pad=0.8)
save(fig, "duration_distribution.pdf")

# ── 3. FIGURE 2 — Composer duration ──────────────────────────────────────────
composer_h = (df.groupby("canonical_composer")["duration"]
               .sum()
               .sort_values(ascending=False)
               / 3600)
top15 = composer_h.head(10)

fig, ax = plt.subplots(figsize=(5.5, 2.3))
y_pos = list(range(len(top15)))
# Reverse so longest bar appears at the top
vals   = top15.values[::-1]
labels = [c.split(",")[0] for c in top15.index[::-1]]

bars = ax.barh(y_pos, vals, color=MAIN_COLOR,
               edgecolor="white", linewidth=0.4, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Total duration (h)")
ax.set_title("Top 10 composers by recording duration")

for bar, val in zip(bars, vals):
    ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}h", va="center", fontsize=7, color="#333333")

fig.tight_layout(pad=0.8)
save(fig, "composer_duration.pdf")

# ── 4. MIDI PARSING ───────────────────────────────────────────────────────────
print(f"\n  Parsing MIDI files (up to {MAX_MIDI_FILES})...")

midi_files = df["midi_filename"].tolist()
if MAX_MIDI_FILES:
    rng = np.random.default_rng(42)
    midi_files = rng.choice(midi_files,
                            size=min(MAX_MIDI_FILES, len(midi_files)),
                            replace=False).tolist()

note_counts, note_densities = [], []
velocities, polyphonies      = [], []
pitch_counts = np.zeros(128)

for fname in midi_files:
    fpath = os.path.join(MAESTRO_DIR, fname)
    try:
        pm    = pretty_midi.PrettyMIDI(fpath)
        notes = [n for inst in pm.instruments for n in inst.notes]
        if not notes:
            continue
        pitches = [n.pitch    for n in notes]
        vels    = [n.velocity for n in notes]
        dur     = pm.get_end_time()

        note_counts.append(len(notes))
        note_densities.append(len(notes) / dur if dur > 0 else 0)
        velocities.extend(vels)

        for p in pitches:
            pitch_counts[p] += 1

        times = np.arange(0, dur, 0.1)
        poly  = [sum(n.start <= t < n.end for n in notes) for t in times]
        polyphonies.append(np.mean(poly))

    except Exception:
        pass

print(f"  Parsed {len(note_counts)} MIDI files successfully.")

# ── 5. FIGURE 3 — MIDI statistics (6-panel) ──────────────────────────────────
fig = plt.figure(figsize=(7.0, 5.0))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.68, wspace=0.44)

panels = [
    (note_counts,    "Note count",             "Notes per piece"),
    (note_densities, "Note density (notes/s)", "Notes per second"),
    (velocities,     "Velocity",               "MIDI velocity"),
    (polyphonies,    "Polyphony",              "Sim.\ notes (mean)"),
]

for idx, (data, title, xlabel) in enumerate(panels):
    row, col = divmod(idx, 2)
    ax = fig.add_subplot(gs[row, col])
    ax.hist(data, bins=30, color=MAIN_COLOR,
            edgecolor="white", linewidth=0.4, alpha=0.85)
    med = np.median(data)
    ax.axvline(med, color=ACCENT, linestyle="--", linewidth=0.9)
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    # Median annotation in top-right corner
    ax.text(0.96, 0.92,
            r"$\tilde{x}=" + f"{med:.1f}" + r"$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=ACCENT)

save(fig, "midi_statistics.pdf")

# ── 6. FIGURE 4 — Pitch usage ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 2.4))

ax.bar(range(128), pitch_counts, width=1.0,
       color=MAIN_COLOR, edgecolor=MAIN_COLOR, linewidth=0.0, alpha=0.85)

# Highlight standard piano range
ax.axvspan(21, 108, alpha=0.07, color="#4DAC26", zorder=0,
           label="Piano range (A0--C8)")
ax.axvline(21,  color="#4DAC26", linestyle=":", linewidth=0.9)
ax.axvline(108, color="#4DAC26", linestyle=":", linewidth=0.9)

# Landmark pitch labels
ymax = pitch_counts.max()
for midi_p, label in [(21, "A0"), (60, "C4"), (69, "A4"), (108, "C8")]:
    ax.axvline(midi_p, color="#AAAAAA", linestyle=":", linewidth=0.7)
    ax.text(midi_p, ymax * 1.03, label,
            ha="center", va="bottom", fontsize=7, color="#555555")

ax.set_xlabel("MIDI pitch")
ax.set_ylabel("Occurrences")
ax.set_title("Pitch usage across MAESTRO dataset", pad=12)
ax.set_xlim(0, 127)
ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
ax.legend(frameon=False, loc="upper center",
          bbox_to_anchor=(0.8, -0.18), ncol=1, fontsize=8)

fig.tight_layout(pad=0.8)
save(fig, "pitch_usage.pdf")

# ── 7. FIGURE 5 — Temporal split distribution ─────────────────────────────────
df["year"] = df["audio_filename"].str.extract(r"(\d{4})/").astype(int)

years  = sorted(df["year"].unique())
splits = ["train", "validation", "test"]
width  = 0.25
x      = np.arange(len(years))

fig, ax = plt.subplots(figsize=(5.5, 2.6))

for i, split in enumerate(splits):
    counts = df[df["split"] == split].groupby("year").size().reindex(years, fill_value=0)
    ax.bar(x + (i - 1) * width, counts.values, width,
           color=SPLIT_COLORS[split], edgecolor="white",
           linewidth=0.4, alpha=0.85, label=split.capitalize())

ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Competition year")
ax.set_ylabel("Recordings")
ax.set_title("Temporal split distribution")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1),
          borderaxespad=0, fontsize=8)

fig.tight_layout(pad=0.8)
save(fig, "temporal_split.pdf")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\nSummary statistics:")
print(f"  Total recordings   : {len(df)}")
print(f"  Total duration     : {df['duration'].sum()/3600:.2f} h")
print(f"  Median piece dur.  : {dur_min.median():.2f} min")
if note_densities:
    print(f"  Median note density: {np.median(note_densities):.2f} notes/s")
    print(f"  Median polyphony   : {np.median(polyphonies):.2f} notes")
    print(f"  Mean velocity      : {np.mean(velocities):.1f}")
print(f"\nFigures saved to: {OUTPUT_DIR}/")
print("Upload to Overleaf under figures/")