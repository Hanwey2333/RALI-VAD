# Data preparation

RALI-VAD is evaluated on UCF-Crime, XD-Violence, and UBnormal. This repository
does not redistribute the original videos, extracted CLIP/I3D features,
frame-level annotations, or model checkpoints.

Users should obtain each dataset from its official provider and comply with the
provider's access and reuse terms. UBnormal is distributed under CC BY-NC-ND
4.0; the terms displayed by the UCF-Crime and XD-Violence providers apply to
those datasets.

The loaders consume JSON manifests that point to locally stored feature arrays.
Keep local absolute paths, credentials, and generated semantic archives outside
version control. The recommended ignored locations are `data/`, `features/`,
`outputs/`, `runs/`, and `checkpoints/`.
