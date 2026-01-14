"""
Basic logging helpers
"""
from omegaconf import OmegaConf, DictConfig, ListConfig
from omegaconf.errors import InterpolationResolutionError

from rich import print as rich_print
import rich.syntax
import rich.tree


def print_header(x, border="both") -> None:
    """Print with borders"""
    match border:
        case "both":
            prefix = f"{"-" * len(x)}\n"
            suffix = f"\n{"-" * len(x)}"
        case "top":
            prefix = f"{"-" * len(x)}\n"
            suffix = ""
        case "bottom":
            prefix = ""
            suffix = f"\n{"-" * len(x)}"
        case _:
            raise ValueError(f"Invalid border: {border}")
    rich_print(f"{prefix}{x}{suffix}")


def print_config(config: DictConfig, name: str = "CONFIG", style="bright") -> None:
    """
    Prints content of DictConfig using Rich library and its tree structure.
    """
    tree = rich.tree.Tree(name, style=style, guide_style=style)
    fields = config.keys()
    for field in fields:
        try:
            branch = tree.add(str(field), style=style, guide_style=style)
            config_section = config.get(field)
            branch_content = str(config_section)
            if isinstance(config_section, DictConfig):
                branch_content = OmegaConf.to_yaml(config_section, resolve=True)
            elif isinstance(config_section, ListConfig):
                branch_content = OmegaConf.to_yaml(config_section, resolve=True)
            branch.add(rich.syntax.Syntax(branch_content, "yaml"))
        
        # except InterpolationResolutionError as e:
        except Exception as e:
            print(f"-> Error resolving interpolation: {e}")
            print(f"-> Field: {field}")
            print(f"-> Config section: {config_section}")
        
    rich.print(tree)
