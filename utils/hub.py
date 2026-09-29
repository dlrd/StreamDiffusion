"""Local-first model loading: a model already in the cache never touches the network."""
import logging


def _describe(args, kwargs) -> str:
    for value in (kwargs.get("repo_id"), kwargs.get("pretrained_model_name_or_path"),
                  args[0] if args and isinstance(args[0], str) else None,
                  args[1] if len(args) > 1 and isinstance(args[1], str) else None):
        if value:
            name = str(value)
            if kwargs.get("filename"):
                name += f"/{kwargs['filename']}"
            elif kwargs.get("subfolder"):
                name += f"/{kwargs['subfolder']}"
            return name
    return "model"


def local_first(load, *args, **kwargs):
    """load(*args, **kwargs) from the local cache, downloading only what is missing.

    Offline, a plain load retries each Hub request 5 times (~23 s per file) before using the cache."""
    if "local_files_only" in kwargs:
        return load(*args, **kwargs)
    try:
        return load(*args, local_files_only=True, **kwargs)
    except (OSError, ValueError) as e:
        logging.info(f"[Hub] {_describe(args, kwargs)} not in the local cache, downloading "
                      f"({type(e).__name__})")
        return load(*args, **kwargs)
