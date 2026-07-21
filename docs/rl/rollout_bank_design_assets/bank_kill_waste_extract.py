# Reproduces the G/G and B/B rows of "Measured cost of the status quo"
# (persisting-rollout-state-design.md) from W&B alone.
#
#   uv run --no-project --with wandb --with pandas --with pyarrow \
#       python bank_kill_waste_extract.py
#
# Requires a W&B API key for adlr/megatron-rl in ~/.netrc.
#
# METHOD
# ------
# Runs: the lag-5 siblings of the 2026-07-17 `lagshape_nvls32ap` sweep
# (8 nodes / 32 GPUs, 64 prompts x group 16, gate = (5+1)*64*16 = 6,144):
#   G/G mkxx5cim, B/B k9wstonf  (R/G bl8qgebf and R/B rmunkfhb are the same
#   sweep; see the R-row note at the bottom).
#
# 1. Full-fidelity history comes from each run's `wandb-history` parquet
#    artifact (the sampled history API returns too few rows, and
#    scan_history rejects these runs' custom step metric).
# 2. Kill boundaries: every restart rebuilds the rank-0 rollout pipeline, so
#    the cumulative per-env `*_pipeline_inferred_count` counters reset to
#    zero. A drop of >64 rollouts in the summed counter marks a new SLURM
#    segment. (Timestamp-gap detection fails for these runs: iterations take
#    >15 min, and both runs also crash-looped, restarting far more often
#    than the 4h limit alone would cause.)
# 3. Per segment, the last logged row is the kill snapshot. Buckets, summed
#    over envs (all counts in rollouts; GROUP = 16 rollouts per group):
#      banked complete   = (output_queue_size + consume_pending_groups) * 16
#      partial members   = assemble_pending_groups * 16/2   (half-filled assumption)
#      engine in-flight  = prepared_count - inferred_count
#        engine active   = min(in-flight, 2784)  # 4 engine DP ranks x 696 slots
#        engine waiting  = remainder (no GPU work done yet -> no loss)
#      trained           = yielded_count * 16
# 4. "Mature" kills only (summed inferred_count >= 3,000 rollouts) enter the
#    averages; shorter segments are crash-loop noise that would skew shares.
# 5. Tokens: rollout counts x the run-wide MEDIAN of `length/traj/mean`
#    (~11.9-12.5k tok; per-segment last values are unreliable on short
#    segments), with mid-decode rollouts counted at half length. Token shares are of the segment's own generation
#    (inferred + act/2).
#
# R-ROW PROVENANCE (not reproduced here): the R/G and R/B rows were produced
# by the earlier analysis with the same queue/flow accounting, but with the
# engine active/waiting split and engine-token footprint measured from the
# `rl_log_inference_batch_trace` per-rank JSONLs on the cluster
# (<run_dir>/rl_logging/inference/), which log active/waiting request counts
# and KV-block usage at every suspend boundary. Those files are only
# reachable from hsg2. This script's engine split is the slot-cap
# approximation described above.

import collections
import glob

import pandas as pd
import wandb

GROUP = 16
ENGINE_SLOTS = 2784           # 4 engine DP ranks x 696 request slots
MATURE_MIN_INFERRED = 3000    # rollouts; below this = crash-loop segment
RESET_DROP = 64               # rollouts; counter drop that marks a restart
RUNS = {"G/G": "mkxx5cim", "B/B": "k9wstonf"}
FIELDS = ["prepared_count", "inferred_count", "assembled_count", "yielded_count",
          "output_queue_size", "consume_pending_groups", "assemble_pending_groups"]

api = wandb.Api(timeout=60)

for label, rid in RUNS.items():
    run = api.run(f"adlr/megatron-rl/{rid}")
    envs = sorted({k.split("_pipeline_")[0] for k in run.summary_metrics if "_pipeline_" in k})

    art = next(a for a in run.logged_artifacts() if a.type == "wandb-history")
    root = art.download(root=f"/tmp/hist_{rid}")
    df = pd.concat([pd.read_parquet(f) for f in
                    glob.glob(f"{root}/**/*.parquet", recursive=True)], ignore_index=True)

    inf_cols = [f"{e}_pipeline_inferred_count" for e in envs
                if f"{e}_pipeline_inferred_count" in df.columns]
    df["_totinf"] = df[inf_cols].sum(axis=1)
    df = df.sort_values("_timestamp").reset_index(drop=True)
    seg_id = (df["_totinf"].diff() < -RESET_DROP).cumsum()

    kills = []
    for _, seg in df.groupby(seg_id):
        seg = seg.ffill()
        last = seg.iloc[-1]
        g = lambda e, f: (last.get(f"{e}_pipeline_{f}") or 0)
        s = collections.Counter()
        for e in envs:
            s.update({
                "outq_g": g(e, "output_queue_size"),
                "reorder_g": g(e, "consume_pending_groups"),
                "partial_m": int(g(e, "assemble_pending_groups")) * GROUP // 2,
                "inflight": max(g(e, "prepared_count") - g(e, "inferred_count"), 0),
                "inferred": g(e, "inferred_count"),
                "yielded_g": g(e, "yielded_count"),
            })
        kills.append(dict(
            outq=int(s["outq_g"] * GROUP), reorder=int(s["reorder_g"] * GROUP),
            partial=s["partial_m"],
            act=int(min(s["inflight"], ENGINE_SLOTS)),
            wait=int(max(s["inflight"] - ENGINE_SLOTS, 0)),
            inferred=int(s["inferred"]), trained=int(s["yielded_g"] * GROUP),
        ))

    mature = [k for k in kills if k["inferred"] >= MATURE_MIN_INFERRED]
    n = len(mature)
    avg = lambda f: sum(k[f] for k in mature) / n
    gen_eq = avg("inferred") + 0.5 * avg("act")           # rollout-equivalents generated
    mt = float(df["length/traj/mean"].dropna().median())
    lost = avg("outq") + avg("reorder") + avg("partial") + avg("act") + avg("wait")

    print(f"\n===== {label} ({rid}): {len(kills)} restarts, {n} mature kills, mean traj ~{mt:,.0f} tok")
    print(f"  per kill: output_queue={avg('outq'):,.0f}  reorder={avg('reorder'):,.0f}  "
          f"partial={avg('partial'):,.0f}  engine_active={avg('act'):,.0f}  waiting={avg('wait'):,.0f}"
          f"  -> total lost {lost:,.0f} rollouts")
    print(f"  token shares of segment generation: "
          f"output_queue={100*avg('outq')/gen_eq:.1f}%  reorder={100*avg('reorder')/gen_eq:.1f}%  "
          f"partial={100*avg('partial')/gen_eq:.1f}%  engine_active={100*0.5*avg('act')/gen_eq:.1f}%  "
          f"| trained={100*avg('trained')/gen_eq:.1f}%")
    print(f"  est tokens: generated/segment ~{gen_eq*mt/1e6:.0f}M, lost/kill ~"
          f"{(avg('outq')+avg('reorder')+avg('partial')+0.5*avg('act'))*mt/1e6:.0f}M")
