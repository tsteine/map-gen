- The config JSON format is still under development. It is ok to make
  backward-incompatible changes to it. Every field should have a simple,
  consistent meaning. If a config field is missing it should give an error: do
  not try to set up default values or fallback behaviors if a field is missing.
  This also applies to Python constructor/function parameters that are derived
  from config fields: keep them required instead of adding default values such
  as `None`, numeric fallbacks, or `x or default` handling.
- Maintain clean, readable code. When there is an opportunity to improve code
  quality by refactoring, suggest this, even if not required for the current
  task.
- Do not bother removing or clearing `python/__pycache__`.
