---
name: Project VM and workflow context
description: Transformer trained on VM, not locally. Local repo is edit-only; runs happen on remote.
type: project
---

Training and inference runs happen on a remote VM at:
`/mnt/data/gharatayush27/projects/cs4644_sentiment_analysis_feature_in_transformers/`

**Why:** Transformer/SAE training requires GPU; local machine is edit-only.

**How to apply:** Any script that needs checkpoints must be run on the VM. Local results (JSON files, loss curves) are synced copies. When suggesting "run X", always mean "push code, then run on VM."
