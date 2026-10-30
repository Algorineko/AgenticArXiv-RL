# Pinned official BFCL native-backend preflight

Official source: https://github.com/ShishirPatil/gorilla at commit 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8. The current official data is labeled BFCL_v4_multi_turn_base; this is not the old Fireworks three-column v3-derived Parquet.

The adapter checks source hashes, loads original initial_config, retains all gold turns, and dispatches literal positional/keyword calls to whitelisted public GorillaFileSystem methods. It does not eval model text, instantiate external messaging/network services, or operate on host files. Native virtual tree transitions and observation hashes are recorded.

25 filesystem-only trajectories / 142 calls replay without Python exceptions; 24 root states change. Mixed/unsupported trajectories are excluded by function support before any model inference. Read-only responses are not established by state equality alone.

This is gold/backend preflight only. No model BFCL score is claimed; full native state, response and path-checker integration is still required. The pinned official source and data remain in local cache rather than being vendored into this repository. Use scripts/rescue_bfcl_native_preflight.py with --source-root pointing to the hash-matched source tree.
