"""Dynamic variant knowledge-base generation utilities."""

__all__ = ["build_dynamic_knowledge_base", "get_source_spec", "list_source_specs"]


def build_dynamic_knowledge_base(*args, **kwargs):
    """Lazily import the pandas-backed builder."""
    from .orchestrator import build_dynamic_knowledge_base as _builder

    return _builder(*args, **kwargs)


def get_source_spec(*args, **kwargs):
    """Lazily import registry lookup."""
    from .registry import get_source_spec as _get_source_spec

    return _get_source_spec(*args, **kwargs)


def list_source_specs(*args, **kwargs):
    """Lazily import registry listing."""
    from .registry import list_source_specs as _list_source_specs

    return _list_source_specs(*args, **kwargs)
