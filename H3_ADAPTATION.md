# MiniMax H3 compiler source adaptation

This branch publishes the preserved compiler sources from the H3 head56
integration snapshot dated 2026-10-23. It is based on upstream NineToothed
v0.25.0, commit `897f97be6a47618c7a251a87435f44880905bec5`.
`h3-source-manifest.json` records SHA256 identities for all 20 preserved files.
The original runtime did not record a compiler Git commit; this publication
establishes a source identity without inventing an earlier revision.

## Changes compared with upstream v0.25.0

* `aot.py`: HIP host compilation and error handling; explicit compiler Python;
  shape/stride specialization and bounded int32 variants with an int64 fallback
  in the general policy; launch argument ABI handling; optional cache policies.
* `build.py`: source and configuration fingerprints to reject stale build caches.
* `generation.py`: `NINETOOTHED_CACHE_DIR` permits a private writable cache.

The other 17 preserved files match upstream after line-ending normalization.
Upstream source history and Apache-2.0 license remain included.

## Reproduction scope

The selected InfiniCore/InfiniLM build compiles host wrappers and links the
embedded generated device code supplied in those repositories. It does not
regenerate every device kernel with this compiler and does not need a new Triton
installation solely to perform that build. Kernel regeneration requires a
separately verified matching Hygon Triton toolchain. Publishing this branch does
not establish that arbitrary compiler policies, shapes, devices, or inputs pass.

No model weights, vendor binary wheels, credentials, or private environments are
included. Later VAE Attention/SwiGLU migration experiments are not enabled by
publishing these compiler sources.
