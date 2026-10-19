# Repository workflow

- The user requested that future changes to this project be committed to this
  repository. After completing and verifying a coherent change, commit the
  relevant source, tests, configuration, and documentation. Use the user's
  configured Git identity; do not invent an author or switch to a bot identity.
- Use concise commit messages, usually a short one-line subject.
- Keep README.md concise: project overview, completed work, measured results,
  and setup/run instructions. Put dated progress logs and detailed experiment
  history in RESEARCH.md or ignored reports; do not prepend them to README.md.
- Keep datasets, checkpoints, generated reports, environments, caches, logs,
  and credentials out of Git. Follow `.gitignore` and inspect staged files
  before committing. Never force-add these artifacts or commit credentials.
- Preserve experiment evidence and running workers. Source files referenced by
  active experiments or closed source bindings must remain unchanged; create
  a new version when a fix is needed. Do not restart or alter live training
  processes as part of repository maintenance.
- Keep commits focused on the requested work. Do not include unrelated local
  changes. GitHub publishing must use the user's authenticated account and the
  repository visibility they selected. Do not force-push or rewrite shared
  history without an explicit request.
