# Package API

Rules for the supported import surface. Read this before adding or renaming a public name, or before changing a provider-visible schema.

- `aashir.agent` and `aashir.tools` are the supported package import surfaces. Re-export supported names in each package's `__init__.py` and list them explicitly in `__all__`.
- Examples and tests that exercise supported public behavior import from these package facades, not from implementation modules. A direct import from an implementation module is not a supported public API.
- Prefix module-level implementation-only helpers, classes, and constants with `_`. Internal modules and focused unit tests may import these names directly. That import does not make them public.
- Keep CLI entry points and framework hooks under the names their callers require. They are entry points, not general package API.
- Pydantic models used in provider-visible JSON Schema keep public, unprefixed class names. Add a one-line comment saying why: Pydantic derives `$defs` and `$ref` keys from the class name, and the provider sees those keys. Leave these models out of `__all__` unless they are supported API. Test the generated schema when renaming one.
