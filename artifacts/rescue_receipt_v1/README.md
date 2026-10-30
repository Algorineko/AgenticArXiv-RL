# Static request-contract and operation-receipt baseline

Pre-inference protocol commit: `767e016`; manifest hash: `be0c82c502b8502f1d2790de33d7f9098f8fbb756588d30aaca9edd212bf9752`.

See docs/research/rescue_receipt_v1_protocol.md. These are offline stub operation receipts. They do not prove real PDF/translation quality. Request compilation only supports explicit numbered single translation or force=true download; unsupported requests use unchanged model prompts. The deterministic request_executor baseline receives user text only, no benchmark gold.

The development recipe has one manually fixed candidate. Its model comparison uses the same checkpoints, machines, call budgets and decoding within each model. Any promotion evaluation is authorized by the predeclared no-regression/positive-gain development rule, never final-test scores.
