Shared paired modes — 2026-10-07

Run with E:\New_research\.conda-envs\mid-py310\python.exe run_all.py
Existing completed training fits are reused. All old research assets are preserved.

joint_model.py: small shared-mode camera/person residual heads and joint scoring.
experiment.py: frozen GRU/flow backbones, full-camera pretraining and joint adaptation.
physical_inference.py: production inference with float64 exact projective geometry.
  Use its infer() function for new evaluation. experiment.infer() retains the
  training's stabilized denominator for provenance and is NOT the final evaluator.
audit.py: selected-output replay, physical future truncation, independent SciPy
  geometry, old attention checkpoint replays, branch/cancellation/selection metrics.
render.py: old fixed nine-case videos, summary and intermediate-branch plots.

Protocol and weights: outputs/joint_modes_20261007/
Report: reports/joint_modes_20261007/RESULTS.txt
Presentation: outputs/joint_modes_20261007/presentation/START_HERE.txt

Three seeds, six original recording folds, 117 windows. K1 is an extra-training
control; comparing K3 with old attention models is a system comparison, not a
capacity/compute-matched attention ablation. Original five flow samples are
averaged only to construct the frozen base; new K3 futures are jointly selected.
Scores are not calibrated probabilities. Oracle selection uses future references
and is strictly diagnostic. The original baseline is not replaced.

Numerical audit notes:
1. CUDA atomic group-logit sums may vary at float32 roundoff. Individual logits,
   candidates, selected mode and final predictions replay exactly.
2. Initial stabilized projections are retained in *_initial_stabilized.npz.
   Exact physical finalization changes no weights, candidate rotations/residuals,
   scores or selected mode. Negative-depth rays are not repaired; metrics penalize
   them explicitly. No selected evaluation ray was invalid in this run.
