"""Adapting Without Forgetting — interactive demo (CS 725).

Everything shown here is read from results we already computed (one seed): per-image predictions of three
final models and the accuracy matrices / curves of the 36 study runs. Nothing is trained live.
Run:  streamlit run app/app.py
"""
import io
import json
import os
import pickle
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

st.set_page_config(page_title="Adapting Without Forgetting", page_icon="🧭", layout="wide")

DOM_NAMES = {
    "pacs": {"P": "Photo", "A": "Art", "C": "Cartoon", "S": "Sketch"},
    "officehome": {"R": "Real World", "A": "Art", "C": "Clipart", "P": "Product"},
}
DS_NAMES = {"pacs": "PACS (7 classes)", "officehome": "Office-Home (65 classes)"}
MODELS = [("source", "Source-only"), ("seq_bn", "Sequential (BN-affine)"), ("replay_bn", "Replay (BN-affine)")]
COL = {"source": "#6b6b6b", "seq_bn": "#D55E00", "replay_bn": "#0072B2"}
SCOPE_NAMES = {"none": "none (source only)", "bn": "BN-affine", "head": "head only", "full": "full encoder",
               "lastblock": "last block"}
METHOD_NAMES = {"fixed": "Fixed (source-only)", "seq": "Sequential pseudo-labelling", "replay": "Sequential + replay memory"}


# --------------------------------------------------------------------------- data
@st.cache_data
def load_preds():
    return pickle.load(open(os.path.join(DATA, "preds.pkl"), "rb"))


@st.cache_data
def load_thumbs(ds, dom):
    return pickle.load(open(os.path.join(DATA, f"thumbs_{ds}_{dom}.pkl"), "rb"))


@st.cache_data
def load_runs():
    d = json.load(open(os.path.join(DATA, "runs.json")))
    return d["runs"], d["diag"]


PREDS = load_preds()
RUNS, DIAG = load_runs()

st.title("Adapting Without Forgetting")
st.caption("Continual unsupervised domain adaptation with a tiny replay memory · ResNet-18 · one seed · "
           "all numbers are precomputed from our experiments")

tab1, tab2, tab3, tab4 = st.tabs(["1 · Try the models", "2 · Change the parameters", "3 · What the memory stores",
                                  "4 · Why replay works"])

# =========================================================================== tab 1
with tab1:
    st.markdown("Pick a domain and look at real test images. Under each image: what the **source-only** model, "
                "the **sequentially adapted** model and the **replay** model predict after the whole stream "
                "(PACS: Photo→Art→Cartoon→Sketch, Office-Home: Real World→Art→Clipart→Product; BN-affine scope).")
    c1, c2, c3 = st.columns(3)
    ds = c1.selectbox("Dataset", list(DS_NAMES), format_func=DS_NAMES.get, key="t1ds")
    dom = c2.selectbox("Domain", list(DOM_NAMES[ds]), format_func=DOM_NAMES[ds].get, key="t1dom")
    T = load_thumbs(ds, dom)
    classes = T["classes"]
    cls_pick = c3.selectbox("Class", ["(any)"] + [c.replace("_", " ") for c in classes], key="t1cls")
    P = PREDS[ds][dom]
    y = P["y"]

    # accuracy of the three models on this domain (and class)
    mask = np.ones(len(y), bool)
    if cls_pick != "(any)":
        mask = y == [c.replace("_", " ") for c in classes].index(cls_pick)
    acc = {k: float((P[k]["pred"][mask] == y[mask]).mean() * 100) for k, _ in MODELS}
    cols = st.columns(3)
    for col, (k, lab) in zip(cols, MODELS):
        col.metric(lab, f"{acc[k]:.1f}%", help=f"accuracy on {mask.sum()} test images of this selection")

    o1, o2, o3 = st.columns([2, 2, 1])
    n_show = o1.slider("Images to show", 4, 16, 8, step=4, key="t1n")
    only_dis = o2.checkbox("Only images where the models disagree", value=False, key="t1dis")
    seed = st.session_state.get("t1seed", 0)
    if o3.button("🔀 Shuffle"):
        seed = random.randint(0, 10**6)
        st.session_state["t1seed"] = seed

    idx = np.where(mask)[0]
    if only_dis:
        pr = np.stack([P[k]["pred"][idx] for k, _ in MODELS])
        idx = idx[~(pr == pr[0]).all(0)]
    if len(idx) == 0:
        st.info("No images match this selection.")
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, min(n_show, len(idx)), replace=False)
        per_row = 4
        for r in range(0, len(idx), per_row):
            cols = st.columns(per_row)
            for col, i in zip(cols, idx[r:r + per_row]):
                col.image(T["jpg"][int(i)], use_container_width=True)
                col.markdown(f"**true: {classes[int(y[i])].replace('_', ' ')}**")
                for k, lab in MODELS:
                    p = int(P[k]["pred"][i])
                    ok = p == int(y[i])
                    conf = float(P[k]["conf"][i]) * 100
                    short = lab.split(" ")[0]
                    col.markdown(f":{'green' if ok else 'red'}[{'✓' if ok else '✗'} {short}: "
                                 f"{classes[p].replace('_', ' ')} ({conf:.0f}%)]")
    st.caption("Reading it: on PACS-Photo the sequential model has forgotten what the source model knew; replay "
               "keeps it. On Sketch (the newest, hardest domain) sequential is better than replay. "
               "On Office-Home all three behave alike.")

# =========================================================================== tab 2
with tab2:
    st.markdown("Choose a configuration. We look up the run with exactly those settings (we ran a fixed grid, so "
                "options that were never run are not offered) and show how the numbers change as you move one control.")
    df = pd.DataFrame([{**r["cfg"], "id": r["id"], **{k: v * 100 for k, v in r["summary"].items()
                       if k in ("acc_final", "forgetting", "new_acc", "src_final", "tgt_final")}} for r in RUNS])

    def opts(frame, col):
        return sorted(frame[col].unique().tolist(), key=lambda v: (str(type(v)), v))

    sel = {}
    f = df
    c1, c2, c3, c4 = st.columns(4)
    sel["dataset"] = c1.selectbox("Dataset", opts(f, "dataset"), format_func=DS_NAMES.get, index=opts(f, "dataset").index("pacs"), key="t2ds")
    f = f[f.dataset == sel["dataset"]]
    sel["method"] = c2.selectbox("Method", opts(f, "method"), format_func=METHOD_NAMES.get, index=opts(f, "method").index("replay"), key="t2m")
    f = f[f.method == sel["method"]]
    sel["scope"] = c3.selectbox("What is updated (scope)", opts(f, "scope"), format_func=SCOPE_NAMES.get,
                                index=opts(f, "scope").index("bn") if "bn" in opts(f, "scope") else 0, key="t2sc")
    f = f[f.scope == sel["scope"]]
    imopts = opts(f, "im_weight")
    sel["im_weight"] = c4.selectbox("Info-max loss weight", imopts, key="t2im") if len(imopts) > 1 else imopts[0]
    f = f[f.im_weight == sel["im_weight"]]
    if sel["method"] == "replay":
        d1, d2, d3, d4 = st.columns(4)
        sel["res"] = d1.select_slider("Memory image size (px)", opts(f, "res"),
                                      value=64 if 64 in opts(f, "res") else opts(f, "res")[0], key="t2res")
        f = f[f.res == sel["res"]]
        rho_opts = opts(f, "rho")
        sel["rho"] = d2.select_slider("Memory budget ρ (% of 224-px data)", rho_opts,
                                      value=0.02 if 0.02 in rho_opts else rho_opts[0],
                                      format_func=lambda v: f"{v * 100:g}%", key="t2rho")
        f = f[f.rho == sel["rho"]]
        mopts = opts(f, "mem")
        sel["mem"] = d3.selectbox("Memory content", mopts, key="t2mem",
                                  format_func=lambda v: {"srctgt": "source + past targets", "tgt": "past targets only (source-free)"}[v]) if len(mopts) > 1 else mopts[0]
        f = f[f.mem == sel["mem"]]
    oopts = opts(f, "order")
    sel["order"] = (st.selectbox("Domain order", oopts, key="t2ord",
                                 format_func=lambda o: " → ".join(DOM_NAMES[sel["dataset"]][c] for c in
                                                                  ([next(iter(DOM_NAMES[sel["dataset"]]))] + list(o))))
                    if len(oopts) > 1 else oopts[0])
    f = f[f.order == sel["order"]]

    row = f.iloc[0]
    run = next(r for r in RUNS if r["id"] == row["id"])
    base = df[(df.dataset == sel["dataset"]) & (df.method == "fixed") & (df.order == sel["order"])]
    base_acc = base.acc_final.iloc[0] if len(base) else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final accuracy (ACC)", f"{row.acc_final:.1f}%",
              None if base_acc is None else f"{row.acc_final - base_acc:+.1f} vs source-only", help="mean over all domains incl. the source, after the last stage")
    m2.metric("Forgetting", f"{row.forgetting:.1f}", help="best earlier accuracy minus final accuracy; lower is better, negative = improved")
    m3.metric("New-domain accuracy", f"{row.new_acc:.1f}%", help="accuracy on each domain right after adapting to it")
    m4.metric("Source accuracy at the end", f"{row.src_final:.1f}%")

    left, right = st.columns([1, 1])
    with left:
        A = np.array(run["acc"]) * 100
        doms = [DOM_NAMES[sel["dataset"]][c] for c in run["domains"]]
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.imshow(A, vmin=0, vmax=100, cmap="Blues")
        ax.set_xticks(range(len(doms)), doms, fontsize=8)
        ax.set_yticks(range(A.shape[0]), ["after source"] + [f"after {d}" for d in doms[1:]], fontsize=8)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center", fontsize=9, color="white" if A[i, j] > 55 else "black")
        ax.set_title("Accuracy matrix: rows = time, columns = test domain", fontsize=9)
        st.pyplot(fig, clear_figure=True)
    with right:
        st.markdown("**Pseudo-labels per stage** (each stage = one target domain, 5 epochs)")
        rows = []
        for si, stg in enumerate(run["stages"]):
            for ei, ep in enumerate(stg["epochs"]):
                rows.append({"step": si * 5 + ei + 1, "stage": DOM_NAMES[sel["dataset"]][stg["domain"]],
                             "kept (coverage)": ep["coverage"] * 100, "kept & correct (PL acc)": ep["pl_acc"] * 100})
        if rows:
            cdf = pd.DataFrame(rows).set_index("step")
            st.line_chart(cdf[["kept (coverage)", "kept & correct (PL acc)"]], height=250)
            st.caption("Coverage = share of target images whose confidence passed the threshold τ. "
                       "PL acc = how many of those pseudo-labels are right (computed with labels for diagnosis only).")
        else:
            st.info("The source-only model does not adapt, so there is nothing to plot.")
        if sel["method"] == "replay":
            mem = run["stages"][-1].get("memory")
            if mem:
                st.markdown(f"**Memory after the last stage:** {mem['n_items']} images stored "
                            f"({mem['bytes_used'] / 1e6:.1f} MB of a {mem['budget_bytes'] / 1e6:.1f} MB budget)")

    # sweep: accuracy vs budget for each resolution
    if sel["method"] == "replay" and sel["dataset"] == "pacs" and sel["scope"] == "bn":
        sw = df[(df.dataset == "pacs") & (df.method == "replay") & (df.scope == "bn") & (df.order == "ACS") &
                (df.mem == "srctgt") & (df.im_weight == 0)]
        fig, ax = plt.subplots(figsize=(7, 2.9))
        for rs, g in sw.groupby("res"):
            g = g.sort_values("rho")
            ax.plot(g.rho * 100, g.acc_final, "o-", label=f"{rs} px", lw=1.6, ms=4)
        seqrow = df[(df.dataset == "pacs") & (df.method == "seq") & (df.scope == "bn") & (df.order == "ACS") & (df.im_weight == 0)]
        ax.axhline(seqrow.acc_final.iloc[0], color=COL["seq_bn"], ls="--", lw=1.2, label="no memory (seq)")
        ax.axhline(base_acc, color=COL["source"], ls=":", lw=1.2, label="source-only")
        ax.scatter([sel["rho"] * 100], [row.acc_final], s=160, facecolors="none", edgecolors="k", lw=2, zorder=5, label="selected")
        ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 5], ["0.5%", "1%", "2%", "5%"])
        ax.set_xlabel("memory budget ρ"); ax.set_ylabel("final ACC (%)"); ax.set_ylim(20, 75)
        ax.legend(fontsize=7, ncol=3, frameon=False); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, clear_figure=True)
        st.caption("Any memory lifts PACS from 27 to about 58–67, but resolution and budget make no consistent "
                   "difference (spread ≈ ±3, one seed). Drag the sliders and watch the circle move.")

# =========================================================================== tab 3
with tab3:
    st.markdown("The memory has a **byte budget**: B = ρ · (images seen) · 3·224² bytes. A smaller stored image "
                "costs fewer bytes, so more of them fit: about (224 / size)² times as many.")
    c1, c2, c3 = st.columns(3)
    ds3 = c1.selectbox("Dataset", list(DS_NAMES), format_func=DS_NAMES.get, key="t3ds")
    rho = c2.select_slider("Budget ρ", [0.005, 0.01, 0.02, 0.05], value=0.02, format_func=lambda v: f"{v * 100:g}%", key="t3rho")
    stage = c3.selectbox("After stage", [1, 2, 3], index=2, key="t3st")
    ref = next(r for r in RUNS if r["cfg"]["dataset"] == ds3 and r["cfg"]["method"] == "replay" and r["cfg"]["scope"] == "bn"
               and r["cfg"]["res"] == 64 and r["cfg"]["rho"] == 0.02 and r["cfg"]["mem"] == "srctgt" and r["cfg"]["im_weight"] == 0
               and r["cfg"]["order"] == ("ACS" if ds3 == "pacs" else "ACP"))
    seen = ref["stages"][stage - 1]["memory"]["seen"]
    st.write(f"Images seen so far (source + target train sets up to this stage): **{seen}**  → budget "
             f"**{rho * seen * 3 * 224 * 224 / 1e6:.1f} MB**")
    d0 = list(DOM_NAMES[ds3])[0 if ds3 == "pacs" else 1]
    T3 = load_thumbs(ds3, d0)
    s3 = st.session_state.get("t3seed", 1)
    if st.button("🔀 Another image", key="t3b"):
        s3 = random.randint(0, 10**6)
        st.session_state["t3seed"] = s3
    i3 = int(np.random.default_rng(s3).integers(len(T3["jpg"])))
    img = Image.open(io.BytesIO(T3["jpg"][i3])).convert("RGB")
    cols = st.columns(4)
    for col, r_ in zip(cols, (224, 112, 64, 32)):
        small = img.resize((r_, r_), Image.BILINEAR)
        n_fit = int(rho * seen * (224 / r_) ** 2)
        col.image(small.resize((128, 128), Image.NEAREST), caption=f"{r_}×{r_} px")
        col.markdown(f"**{min(n_fit, seen):,} images** fit" + (" (all of them)" if n_fit >= seen else ""))
    mem = ref["stages"][stage - 1]["memory"]
    if rho == 0.02:
        st.success(f"Check against our real run (64 px, ρ=2%): the memory actually held **{mem['n_items']}** images "
                   f"after stage {stage}, close to the formula's {int(rho * seen * (224 / 64) ** 2)} (class balancing "
                   f"and the domain split round down).")
    st.caption("Images above are shown enlarged with nearest-neighbour so you can see what each size throws away. "
               "Our finding: even 30 full-size images work as well as thousands of tiny ones.")

# =========================================================================== tab 4
with tab4:
    st.markdown("**Question:** the BN-affine sequential model collapses on Sketch even though its pseudo-labels are "
                "~93–96% right. Is the replay memory working because it *rehearses* old images, or for some other reason? "
                "We switched pieces off one at a time (PACS, Photo→Art→Cartoon→Sketch).")
    P0 = "pacs_resnet18_s0_ACS_"
    study = {r["id"]: r for r in RUNS}
    diag = {r["id"]: r for r in DIAG}
    items = [
        ("Source-only", study[P0 + "fixed_none"], "#6b6b6b"),
        ("Sequential", study[P0 + "seq_bn"], "#D55E00"),
        ("Seq, BN stats\nfrozen", diag[P0 + "seq_bn"], "#E69F00"),
        ("Seq, only stats\nupdate (lr=0)", diag[P0 + "seq_bn_lr0"], "#a33a00"),
        ("Replay, frozen\nstats", diag[P0 + "replay_bn_r64_rho0.02_srctgt"], "#56B4E9"),
        ("Replay, no loss\non memory", diag[P0 + "replay_bn_r64_rho0.02_srctgt_replay_weight0"], "#3a8ac0"),
        ("Replay (full)", study[P0 + "replay_bn_r64_rho0.02_srctgt"], "#0072B2"),
    ]
    vals = [r["summary"]["acc_final"] * 100 for _, r, _ in items]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(range(len(items)), vals, color=[c for _, _, c in items])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.axhline(vals[0], color="#6b6b6b", ls=":", lw=1)
    ax.set_xticks(range(len(items)), [n for n, _, _ in items], fontsize=8)
    ax.set_ylabel("final ACC (%)"); ax.set_ylim(0, 75); ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, clear_figure=True)
    st.markdown(
        "- **Only the statistics update (lr = 0): 22.3.** No weight changes at all, yet it collapses. BatchNorm's running "
        "averages drift to describe Sketch only → that alone is enough to break the model.\n"
        "- **Replay with no loss on the memory images: 57.6.** The memory is only mixed into the batch (so the "
        "statistics stay balanced). That already gives most of the gain.\n"
        "- **Replay with frozen statistics: 52.0 ≈ source-only.** Rehearsal alone adds no adaptation gain.\n"
        "- So the memory mainly works as a **BatchNorm-statistics stabiliser**; the rehearsal loss adds the last ~6 points.")
    st.caption("One seed, one dataset, one order: a hypothesis supported by these runs, not a general result.")
