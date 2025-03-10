"""Manifest generation"""

from fractal_tasks_core.dev.create_manifest import create_manifest

if __name__ == "__main__":
    """Generate JSON schemas for task arguments afresh, and write them
    to the package manifest.
    """
    PACKAGE = "fractal_helper_tasks"
    AUTHORS = "Joel Luethi"
    docs_link = "https://github.com/jluethi/fractal-helper-tasks"
    if docs_link:
        create_manifest(package=PACKAGE, authors=AUTHORS, docs_link=docs_link)
    else:
        create_manifest(package=PACKAGE, authors=AUTHORS)
