# Fixed batched-discovery development control

The first two train probes both exhausted twelve model steps without official task success. Their official references both pass; 24 model-dispatched observations and both final checks replay exactly. Seven or eight of each run's twelve proxy calls were documentation/discovery.

Before further model calls, freeze one task-agnostic static system addition in `configs/rsi/batched_discovery.txt`. It reminds the agent of the existing twelve-response budget and of its existing permission to issue up to four independent tool calls together. It does not add API names, solutions, task-specific hints, new tools, a larger horizon or a different model. The baseline already permitted four parallel tool calls; it chose to issue them serially.

Run both of the same exposed train scenarios once, in the same order, with the same decoding, native checks, public API proxy and hard caps: thirty attempted requests, 200,000 input and 30,000 output tokens, minimum fifteen-second spacing. Keep a separate private run directory and experiment prefix. Preserve original traces and source snapshots. Record the new harness digest and source digests before inference.

This is a post hoc development control suggested by baseline traces. It cannot establish independent transfer, recursive improvement, or a general failure mechanism; one trajectory per condition cannot isolate stochastic API variation. It tests whether a simple fixed harness change deserves inclusion as a healthy baseline before spending on RSI/RL/TTT.
