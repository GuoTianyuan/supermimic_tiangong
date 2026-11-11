2026-11-11: refactor/isaaclab-layout, based on bc_100hz c0bd65f

Scope and source references
- Local official checkout: /home/mig/supermimic_ws/supermimic, 5efbf114.
  Adopted its src package layout and separation of configuration/runner concerns.
- Local Isaac Lab examples: scripts/reinforcement_learning/rsl_rl and source/.
  Adopted thin execution scripts separate from reusable library code.
- This is an organization change, not a simulator/algorithm port: the fixed
  Isaac Lab 3 kit-less/Newton/MuJoCo-Warp backend stays unchanged.

Relocation and entry points
- 62 original source/script/document paths relocated; see path_map.json.
- Public modules such as supermimic.networks, mdp, motion, buffer and robot keep
  their import names. These numerical modules are not copied into duplicate trees.
- Training CLI schema moved to src/supermimic/configs/train.py. All options,
  defaults, saved args and learning settings remain the same.
- Training/evaluation execution modules moved to runners/. Progress formatting
  and CSV/TensorBoard schema moved to runners/progress.py.
- Native and Docker launchers target scripts/training/train.py; process lookup
  accepts both the pre-refactor and new command paths. stop/status logic remains.
- No old training process is stopped, restarted or updated by this local refactor.
- scripts/export keeps both distinct ONNX interfaces, and playback is retained.
- All old tools are preserved at purpose-specific paths. Machine-specific queue,
  dataset staging and capacity helpers belong to their dated experiment.
- Source packaging is editable, --no-deps; requirements-native.lock, vendor and
  asset files remain unchanged. A wheel contains source, not gigabytes of assets.
  For non-editable use set SUPERMIMIC_ROOT to the checkout holding data/.
- .vscode is private ignored IDE state. No logs, checkpoints or robot assets deleted.

Additional display-only defect found during review
- The legacy offline renderer and failure analyzer hardcoded 50 Hz / .02 s.
- New native/MuJoCo recordings store control_dt, and those tools read it.
- Old recordings require --control-hz; conflicting frequency metadata is rejected.
- Rendering samples requested output timestamps even when FPS does not divide
  control Hz. This affects video timing and analysis labels, not policy/WM/reward.
- Failure analysis also bounds its final-frame index for a recording with no termination.
- Capture the capacity helper's frame tensor in its callback before local cleanup.

Verification
- Validation details are recorded in validation.json.
- Numerical module / full training-flow AST snapshots originate from the baseline
  before moving files. Only import/path/UI relocation and recording metadata are
  ignored in the comparison; numerical learning expressions are not ignored.
- CPU checks cover training options, loss/gradient/optimizer equality, replay,
  padding/noise, both WM prediction modes, checkpoint round trips and log schema.
- Source syntax, undefined names, shell syntax, relocated references, package
  installation, import side effects and asset roots are checked.
- --help tests use a minimal Lab CLI stub only; they do not exercise physics.
- Local machine lacks the pinned Isaac Lab/Newton environment and usable NVIDIA
  driver. Full simulator, CUDA performance, multi-rank training and true resume
  integration have not been run here. tests/integration/ keeps those checks.
