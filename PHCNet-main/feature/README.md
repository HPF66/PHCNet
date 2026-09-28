# Local Frozen Features

Precomputed `.pt` files are not distributed with the source repository. Build
them locally from ESM2-8M, ESM2-650M, and ProtBERT by following the **Feature
Caches** section in the root `README.md`.

For all three PLMs, the builder uses 480 residues per chunk and 64 residues of
overlap, then merges overlapping positions by aligned averaging. The complete
stitched sequence is retained; no 512-residue truncation is used. These tensors
are frozen PLM features, not trainable checkpoints. The trainable adapters and
PHCNet heads are defined in `model/`.
