- The config JSON and model checkpoint formats are still under development. It
  is ok to make backward-incompatible changes to them. Every field should have a
  simple, consistent meaning. If a config or checkpoint field is missing it
  should give an error: do not try to set up default values or fallback
  behaviors if a field is missing. This also applies to Python
  constructor/function parameters that are derived from config fields: keep them
  required instead of adding default values such as `None`, numeric fallbacks,
  or `x or default` handling.
- More generally, avoid adding optional/default function parameters as a way to
  preserve old call sites, enable temporary A/B behavior, or hide missing
  required dependencies. If behavior needs to vary, make the choice explicit at
  the call site with required parameters, separate functions, or a small
  required config object. Do not add `None` defaults, optional lock/context
  hooks, boolean mode defaults, or fallback construction inside the callee
  unless the value is genuinely optional domain data.
- Maintain clean, readable code. When there is an opportunity to improve code
  quality by refactoring, suggest this, even if not required for the current
  task.
- For PyO3 bindings, prefer returning named `#[pyclass]` result objects when a
  function needs to return many values. Avoid creating complex or deeply nested
  tuple return types just to stay within PyO3 tuple arity limits.
- Avoid defining functions inside of other function definitions.
- The training script `train.py` may be run on a different machine, so do not
  make assumptions based on the local development environment.
- Activate the "map-gen" conda environment before running Python.
- Do not bother removing or clearing `python/__pycache__`.
- Ask for confirmation before diving into making changes, unless the desired
  change is already clear from the user's request.
