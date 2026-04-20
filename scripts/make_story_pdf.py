#!/usr/bin/env python3
"""Assemble manuscript sections into a single Markdown file and render to PDF via pandoc.

Produces two PDFs:
  - manuscript-draft.pdf  (lean main text + 5 figures)
  - supplement-draft.pdf  (detailed methods, stats, extended results, 12 figures)
"""

from pathlib import Path
import subprocess, sys

ERSP = Path(__file__).resolve().parent.parent
FIG_DIR = Path("/path/to/your/ERSP_data/Derivatives/minimal/figures")

TITLE = (
    "Frequency-Specific Operant Learning in Neurofeedback Reveals "
    "Distinct Cortical Mechanisms: Evidence from Double-Blind "
    "ERSP and ERP Dissociations"
)
AUTHORS = "Andrew Hill, PhD"
AFFILIATION = "Peak Brain Institute, Los Angeles, CA"

ABSTRACT = r"""
\textbf{Background.}
Neurofeedback reliably alters EEG activity, but the cortical mechanism by which
reward-contingent feedback shapes oscillatory dynamics remains unresolved. In
particular, no study has examined reward-locked event-related spectral
perturbations (ERSP) under double-blind, active-placebo-controlled conditions.

\textbf{Methods.}
Forty participants underwent five training sessions and a 3--5 week retention
session of single-channel EEG biofeedback (C3 SMR 12--15 Hz, $n = 8$; C3 Beta
15--18 Hz, $n = 8$; C4 SMR 12--15 Hz, $n = 8$; active-placebo sham, $n = 16$),
with concurrent 64-channel EEG recording. ERSP was computed from reward-locked
epochs (approximately 600--700 trials per session) using Morlet wavelets (3--40 Hz) across
four sessions.

\textbf{Results.}
Active groups produced frequency-specific event-related desynchronization (ERD) in
the rewarded band (Active vs Sham $d = -1.23$,
$p_{\mathrm{adj}} = 0.001$), absent in sham. A pattern consistent with a double dissociation emerged at C3:
beta training produced the strongest ERD ($d = -2.38$), whereas SMR training
produced the strongest P2 ERP ($d = -1.33$, $\mathrm{BF}_{01} = 0.10$). Only SMR
groups showed lasting plasticity, with increased eyes-closed alpha at follow-up
(C3 SMR $d = 0.97$; C4 SMR $d = 0.78$) and significant across-session
accumulation ($\beta = 1.44$, $p = 0.004$). ERD magnitude predicted long-term
resting-state change ($r = 0.54$, $p = 0.009$) but not within-session shifts
($r = -0.09$, $p = 0.67$), dissociating transient from consolidating effects.
An ICA-based sensitivity analysis confirmed convergence of all primary findings.

\textbf{Conclusions.}
Neurofeedback engages frequency-specific, contingency-dependent cortical
mechanisms. The ERD--P2 dissociation suggests that SMR and beta training
may recruit distinct circuits---potentially thalamocortical versus local cortical---with different
capacities for consolidation. These findings establish a multi-timescale model in
which immediate reward-locked desynchronization drives durable plasticity only when
supported by deeper circuit dynamics.
""".strip()

MAIN_FIGURES = [
    ("ersp_grid_C3.png", "Figure 1",
     "Grand-average ERSP heatmaps at C3 (4 groups $\\times$ 4 sessions). "
     "Active groups show frequency-specific ERD in the reward band; "
     "sham shows theta ERS only."),
    ("frequency_crossover.png", "Figure 2",
     "Frequency crossover at C3. C3 SMR ERD peaks at 12--15 Hz; "
     "C3 Beta peaks at 15--18 Hz; sham is flat."),
    ("erp_p2_focus.png", "Figure 3",
     "Session-averaged reward-evoked ERPs at C3. The P2 (shaded) is "
     "driven by C3 SMR, not C3 Beta---the opposite pattern from the ERD "
     "(full waveforms at C3, C4, Pz in Supplementary Figure S9)."),
    ("erd_resting_scatter.png", "Figure 4",
     "Per-subject ERD during training vs eyes-closed resting-state "
     "change at follow-up (*r* = 0.54, *p* = 0.009, *n* = 22)."),
    ("composite_summary.png", "Figure 5",
     "Multi-panel composite summary of the reward-locked ERSP mechanism."),
]

SUPP_FIGURES = [
    ("erd_learning_curve.png", "Figure S1",
     "Within-session ERD learning curves (early vs late)."),
    ("ersp_grid_C4.png", "Figure S2",
     "Grand-average ERSP heatmaps at C4 (4 groups $\\times$ 4 sessions)."),
    ("topomaps_erd.png", "Figure S3",
     "64-channel topographic ERD maps showing lateralized desynchronization."),
    ("ersp_diff_C3.png", "Figure S4",
     "Active--Sham difference ERSP at C3 with cluster permutation contours (p = 0.002)."),
    ("ersp_diff_C4.png", "Figure S5",
     "Active--Sham difference ERSP at C4 (no clusters survived correction; min p = 0.13)."),
    ("erd_by_session.png", "Figure S6",
     "ERD magnitude by Group $\\times$ Session. The ERD is immediate and stable."),
    ("erd_violins.png", "Figure S7",
     "Individual-subject ERD violin plots with strip overlay."),
    ("frequency_crossover_C4.png", "Figure S8",
     "Frequency crossover at C4 (all 4 groups)."),
    ("erp_waveforms.png", "Figure S9",
     "Full reward-evoked ERP waveforms at C3, C4, Pz (Sessions 1 and 5). "
     "Component windows shaded: P50 (40--80 ms), N1 (80--140 ms), P2 (140--260 ms)."),
    ("resting_growth_curve.png", "Figure S10",
     "Pre-session eyes-closed alpha power at C3 across sessions. "
     "SMR groups show cumulative accumulation; sham and C3 Beta are flat or declining. "
     "Error bars: SEM."),
    ("sensitivity_comparison.png", "Figure S11",
     "Sensitivity analysis: minimal vs ICA preprocessing. "
     "Left: trial retention scatter plot (each point = one subject $\\times$ session); "
     "the dashed line indicates equal retention. "
     "Right: per-group ERD distributions under both pipelines. "
     "The minimal pipeline produces slightly larger effect sizes despite comparable "
     "trial counts, consistent with signal preservation (Delorme, 2023)."),
]


def strip_header(text: str) -> str:
    """Remove front-matter / status blocks from draft files."""
    lines = text.splitlines(keepends=True)
    last_hr = -1
    for i, line in enumerate(lines):
        if i > 20:
            break
        if line.strip() == "---":
            last_hr = i
    start = last_hr + 1 if last_hr >= 0 else 0
    out = []
    for line in lines[start:]:
        if line.startswith("**Source**:") or line.startswith("**Status**:"):
            continue
        if line.startswith("**Conflicts") or line.startswith("**Organized by"):
            continue
        out.append(line)
    while out and out[0].strip() == "":
        out.pop(0)
    return "".join(out)


def read_section(name: str) -> str:
    return strip_header((ERSP / name).read_text())


def fig_block(fname: str, label: str, caption: str) -> str:
    src = FIG_DIR / fname
    if not src.exists():
        return f"\n*[{label}: {fname} not found]*\n\\clearpage\n"
    return (
        f"\n![**{label}.** {caption}]({src}){{width=100%}}\n"
        f"\n\\clearpage\n"
    )


def yaml_block(title: str, author: str, date: str,
               affiliation: str = "") -> str:
    lines = [
        "---",
        f'title: "{title}"',
        f'author: "{author}"',
        f'date: "{date}"',
        "geometry: margin=1in",
        "fontsize: 11pt",
        "linestretch: 1.15",
        "header-includes:",
        "  - \\usepackage{graphicx}",
        "  - \\usepackage{float}",
        "  - \\usepackage{booktabs}",
    ]
    if affiliation:
        lines.append(
            f'  - \\usepackage{{titling}}'
        )
        lines.append(
            f'  - \\preauthor{{\\begin{{center}}\\large}}'
        )
        lines.append(
            f'  - \\postauthor{{\\\\\\normalsize {affiliation}\\end{{center}}}}'
        )
    lines.append("---\n")
    return "\n".join(lines) + "\n"


def extract_intro(text: str) -> str:
    """Pull Part I Introduction paragraphs, stripping ### Paragraph headers."""
    lines = text.splitlines(keepends=True)
    out, in_part1 = [], False
    for line in lines:
        if "## Part I: Introduction" in line:
            in_part1 = True
            continue
        if in_part1 and line.startswith("## Part II"):
            break
        if in_part1:
            if line.startswith("### Paragraph"):
                continue
            out.append(line)
    return "".join(out).strip()


def extract_discussion(text: str) -> str:
    """Pull Part III Discussion."""
    lines = text.splitlines(keepends=True)
    out, in_part3 = [], False
    for line in lines:
        if "## Part III" in line:
            in_part3 = True
            continue
        if in_part3:
            out.append(line)
    return "".join(out).strip()


REFERENCES = r"""
Arnold, L.E., Lofthouse, N., Hersch, S., et al. (2013). EEG neurofeedback for ADHD: Double-blind sham-controlled randomized pilot feasibility trial. *Journal of Attention Disorders*, 17(5), 410--419.

Arnold, L.E., Arns, M., Barterian, J., et al. (2021). Double-blind placebo-controlled randomized clinical trial of neurofeedback for attention-deficit/hyperactivity disorder with 13-month follow-up. *Journal of the American Academy of Child & Adolescent Psychiatry*, 60(7), 841--855.

Arns, M., de Ridder, S., Strehl, U., Breteler, M., & Coenen, A. (2009). Efficacy of neurofeedback treatment in ADHD: the effects on inattention, impulsivity and hyperactivity: a meta-analysis. *Clinical EEG and Neuroscience*, 40(3), 180--189.

Cortese, S., Ferrin, M., Brandeis, D., et al. (2016). Neurofeedback for attention-deficit/hyperactivity disorder: meta-analysis of clinical and neuropsychological outcomes from randomized controlled trials. *Journal of the American Academy of Child & Adolescent Psychiatry*, 55(6), 444--455.

Delorme, A. (2023). EEG is better left alone. *Scientific Reports*, 13, 2372.

Dessy, E., Mairesse, O., van Puyvelde, M., Cortoos, A., Neyt, X., & Pattyn, N. (2020). Train your brain? Can we really selectively train specific EEG frequencies with neurofeedback training. *Frontiers in Human Neuroscience*, 14, 22.

Egner, T., & Gruzelier, J.H. (2001). Learned self-regulation of EEG frequency components affects attention and event-related brain potentials in humans. *NeuroReport*, 12(18), 4155--4159.

Egner, T., & Gruzelier, J.H. (2004). EEG biofeedback of low beta band components: frequency-specific effects on variables of attention and event-related brain potentials. *Clinical Neurophysiology*, 115(1), 131--139.

Fetz, E.E. (1969). Operant conditioning of cortical unit activity. *Science*, 163(3870), 955--958.

Fetz, E.E. (2007). Volitional control of neural activity: implications for brain--computer interfaces. *Journal of Physiology*, 579(3), 571--579.

Gramfort, A., Luessi, M., Larson, E., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.

Grandchamp, R., & Delorme, A. (2011). Single-trial normalization for event-related spectral decomposition reduces sensitivity to noisy trials. *Frontiers in Psychology*, 2, 236.

Hill, A.R. (2012). *Evoked Responses to EEG Biofeedback: A Double-Blind Sham-Controlled Study.* Doctoral dissertation, University of California, Los Angeles.

Hughes, S.W., & Crunelli, V. (2005). Thalamic mechanisms of EEG alpha rhythms and their pathological implications. *The Neuroscientist*, 11(4), 357--372.

Makeig, S. (1993). Auditory event-related dynamics of the EEG spectrum and effects of exposure to tones. *Electroencephalography and Clinical Neurophysiology*, 86(4), 283--293.

Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of EEG- and MEG-data. *Journal of Neuroscience Methods*, 164(1), 177--190.

Muraoka, T., Iwama, S., & Ushiba, J. (2024). Neurofeedback-induced desynchronization of sensorimotor rhythm elicits pre-movement downregulation of intracortical inhibition. *Imaging Neuroscience*.

Pfurtscheller, G., & Lopes da Silva, F.H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110(11), 1842--1857.

Ros, T., Munneke, M.A., Ruge, D., Gruzelier, J.H., & Rothwell, J.C. (2010). Endogenous control of waking brain rhythms induces neuroplasticity in humans. *European Journal of Neuroscience*, 31(4), 770--778.

Ros, T., Enriquez-Geppert, S., Zotev, V., et al. (2020). Consensus on the reporting and experimental design of clinical and cognitive-behavioural neurofeedback studies (CRED-nf checklist). *Brain*, 143(6), 1674--1685.

Salansky, N., et al. (2024). Sensori-motor neurofeedback improves inhibitory control and induces neural changes: a placebo-controlled, double-blind, event-related potentials study. *International Journal of Clinical and Health Psychology*.

Salansky, N., et al. (2025). Enhancing attentional processing through sensorimotor neurofeedback training: evidence from a placebo-controlled, double-blind, event-related potentials study. *NeuroImage*.

Schönenberg, M., Wiedemann, E., Schneidt, A., et al. (2017). Neurofeedback, sham neurofeedback, and cognitive-behavioural group therapy in adults with attention-deficit hyperactivity disorder: a triple-blind, randomised, controlled trial. *The Lancet Psychiatry*, 4(9), 673--684.

Sherlin, L.H., Arns, M., Lubar, J., et al. (2011). Neurofeedback and basic learning theory: implications for research and practice. *Journal of Neurotherapy*, 15(4), 292--304.

Sitaram, R., Ros, T., Stoeckel, L., et al. (2017). Closed-loop brain training: the science of neurofeedback. *Nature Reviews Neuroscience*, 18(2), 86--100.

Sorger, B., Scharnowski, F., Linden, D.E.J., Hampson, M., & Young, K.D. (2019). Control freaks: Towards optimal selection of control conditions for fMRI neurofeedback studies. *NeuroImage*, 186, 256--265.

Sterman, M.B. (1977). Sensorimotor EEG operant conditioning: experimental and clinical effects. *Pavlovian Journal of Biological Science*, 12(2), 63--92.

Sterman, M.B. (2000). Basic concepts and clinical findings in the treatment of seizure disorders with EEG operant conditioning. *Clinical Electroencephalography*, 31(1), 45--55.

Tallon-Baudry, C., Bertrand, O., Delpuech, C., & Pernier, J. (1996). Stimulus specificity of phase-locked and non-phase-locked 40 Hz visual responses in human. *Journal of Neuroscience*, 16(13), 4240--4249.

Thibault, R.T., Lifshitz, M., & Raz, A. (2016). The self-regulating brain and neurofeedback: Experimental science and clinical promise. *Cortex*, 74, 247--261.

Thibault, R.T., & Raz, A. (2017). The psychology of neurofeedback: Clinical intervention even if applied placebo. *American Psychologist*, 72(7), 679--688.

Van Doren, J., Arns, M., Heinrich, H., Vollebregt, M.A., Strehl, U., & Loo, S.K. (2019). Sustained effects of neurofeedback in ADHD: a systematic review and meta-analysis. *European Child & Adolescent Psychiatry*, 28(3), 293--305.
""".strip()


def build_main() -> str:
    parts = [yaml_block(TITLE, AUTHORS, "Draft — April 2026",
                        affiliation=AFFILIATION)]

    parts.append("# Abstract\n\n")
    parts.append(ABSTRACT + "\n\n")
    parts.append("\\newpage\n\n")

    intro_full = read_section("INTRODUCTION_DRAFT.md")
    parts.append("# 1. Introduction\n\n")
    parts.append(extract_intro(intro_full) + "\n\n")

    parts.append("# 2. Methods\n\n")
    parts.append(read_section("METHODS_DRAFT.md").strip() + "\n\n")

    parts.append("# 3. Results\n\n")
    parts.append(read_section("RESULTS_DRAFT.md").strip() + "\n\n")

    parts.append("\\newpage\n\n# Figures\n\n")
    for fname, label, caption in MAIN_FIGURES:
        parts.append(fig_block(fname, label, caption) + "\n")

    parts.append("# 4. Discussion\n\n")
    parts.append(extract_discussion(intro_full) + "\n\n")

    parts.append("\\newpage\n\n# References\n\n")
    parts.append(REFERENCES + "\n\n")

    parts.append("\\newpage\n\n# Acknowledgments\n\n")
    parts.append(
        "Data were collected at the Department of Psychology, "
        "University of California, Los Angeles, under IRB approval. "
        "The author thanks the original dissertation committee and "
        "the participants who contributed their time.\n\n"
    )

    parts.append("# Conflict of Interest\n\n")
    parts.append(
        "Peak Brain Institute, with which the author is affiliated, "
        "provides clinical neurofeedback services. "
        "The data analyzed in this study were collected during the "
        "author's doctoral research at UCLA and are unrelated to "
        "Peak Brain Institute's clinical operations. "
        "This reanalysis was conducted independently without "
        "external funding.\n\n"
    )

    parts.append("# Data and Code Availability\n\n")
    parts.append(
        "Analysis code is available at "
        "https://github.com/DocSalamandyr/Hill\\_2026\\_ERSP-neurofeedback-analysis. "
        "Derived data (ERSP matrices, ERP averages, resting-state PSD, and "
        "statistical outputs) are deposited at https://doi.org/10.5281/zenodo.19555777. "
        "The pre-specified analysis plan is posted to OSF. "
        "Raw EEG recordings (BioSemi BDF) are available from the corresponding "
        "author subject to a data use agreement, as the original informed consent "
        "did not include provisions for unrestricted public sharing.\n\n"
    )

    return "".join(parts)


def build_supplement() -> str:
    parts = [yaml_block(
        "Supplementary Materials",
        AUTHORS,
        "Draft — April 2026",
    )]

    supp = read_section("SUPPLEMENT_DRAFT.md")
    parts.append(supp.strip() + "\n\n")

    parts.append("\\newpage\n\n# Supplementary Figures\n\n")
    for fname, label, caption in SUPP_FIGURES:
        parts.append(fig_block(fname, label, caption) + "\n")

    return "".join(parts)


def render_pdf(md_path: Path, pdf_path: Path):
    cmd = [
        "pandoc", str(md_path),
        "-o", str(pdf_path),
        "--pdf-engine=xelatex",
        "--wrap=auto",
        "-V", "colorlinks=true",
        "-V", "linkcolor=blue",
        "--toc",
        "--toc-depth=3",
    ]
    print(f"  pandoc → {pdf_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[:3000])
        raise RuntimeError(f"pandoc failed with exit code {result.returncode}")
    sz = pdf_path.stat().st_size
    print(f"  {pdf_path.name}: {sz / 1024:.0f} KB")


def main():
    main_md = ERSP / "Hill_2026_ERSP_Neurofeedback_Manuscript.md"
    main_pdf = ERSP / "Hill_2026_ERSP_Neurofeedback_Manuscript.pdf"
    supp_md = ERSP / "Hill_2026_ERSP_Neurofeedback_Supplement.md"
    supp_pdf = ERSP / "Hill_2026_ERSP_Neurofeedback_Supplement.pdf"

    md = build_main()
    main_md.write_text(md)
    print(f"Main manuscript: {len(md.split()):,} words")

    smd = build_supplement()
    supp_md.write_text(smd)
    print(f"Supplement: {len(smd.split()):,} words")

    render_pdf(main_md, main_pdf)
    render_pdf(supp_md, supp_pdf)

    print("Done.")


if __name__ == "__main__":
    main()
