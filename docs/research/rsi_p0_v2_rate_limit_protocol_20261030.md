# P0 v2: bounded continuation after observed RPM limits

P0 v1 stopped after 12 API attempts, of which the last two received explicit RPM HTTP 429 errors. Its three started episodes include two completed successes and one interrupted API episode; this is not a completed model score. It consumed 22,322 conservatively accounted input tokens and 2,958 output tokens, including unknown-usage reservations for the two 429 responses.

Preserve the v1 protocol, traces, summary and source snapshots without altering scores. The independent replay matched all 11 dispatched observations.

Before the next API request, add a minimum seven-second request interval and a single additional thirty-second wait before retrying a 429. This changes transport pacing, not tasks, tools, prompts, model identity, generation settings or scoring. v2 performs the planned two complete passes on all eight development tasks; these remain exposed calibration tasks.

Combined v1+v2 caps remain 100 attempted requests, 300,000 accounted input and 50,000 accounted output tokens. Therefore launch v2 with caps 88 / 277,678 / 47,042. Keep both runs separate and report combined resource usage. A second partial run is not silently merged into a complete score. Each run preserves the source snapshots matching its protocol digests.

No RSI intervention has been introduced. The only empirical observation used to make this change is a provider transport rate limit.
