# Native backend gold replay preflight

Pinned official gorilla commit: 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8.
All 200 original base scenarios, 734 user turns and 1,142 logical gold calls replayed without exceptions. Two independent instances produced matching native state and response checks in all 200 scenarios. These are reference execution checks, not model scores or independent proof of task semantics. All eight APIs are local simulations; no real outgoing messages or external service mutations occur.

Use scripts/fetch_rescue_bfcl_sources.py with the original source_manifest.json, integration_manifest.json and this full_backend_manifest.json, then scripts/rescue_bfcl_full_preflight.py. Sources and manifest hashes are checked before execution; model text is never passed to eval.
