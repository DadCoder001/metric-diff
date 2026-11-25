# Security policy

metric-diff runs locally on arrays you pass it and sends no data anywhere. The only network access is Hugging Face `evaluate` downloading its metric scripts from the Hugging Face Hub the first time each metric is used, and the optional benchmark download script.

If you find a security problem (for example in how files are loaded by the command line), please report it privately using GitHub's **Report a vulnerability** button on the repository's Security tab rather than in a public issue. You'll get a reply within a week.

Supported versions: the latest release.
