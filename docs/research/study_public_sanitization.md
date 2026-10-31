# Study branch publication sanitization

This branch's reachable history was rewritten to remove local machine paths and a legacy host address before publication. Its commit IDs differ from earlier copies, and consumers should clone or rebase onto the rewritten branch instead of merging its old history.

Local paths appearing in historical logs, training examples, and evaluation artifacts were replaced with placeholders. These artifacts remain historical records: file digests and exact text from the earlier unsanitized history may no longer match, so they should not be treated as byte-for-byte reproducible copies. Standard operating-system references and upstream author attribution were retained. The BFCL data path is now resolved relative to the repository; the dataset must be supplied separately when absent.

The sanitization covers known path and credential patterns and is not proof that every possible sensitive value has been identified. The default branch and other branches have independent histories and are outside this branch's cleanup scope.
