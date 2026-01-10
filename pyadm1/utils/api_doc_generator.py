"""
Automatic API Documentation Generator for PyADM1

This module generates comprehensive API reference documentation in Markdown format
for the PyADM1 components package by inspecting Python modules and extracting
docstrings, signatures, and structure.

Example:

    >>> from pyadm1.utils.api_doc_generator import generate_components_api_docs
    >>>
    >>> # Generate documentation for components package
    >>> generate_components_api_docs(
    ...     output_dir="docs/api_reference",
    ...     package_name="pyadm1.components"
    ... )
"""

import inspect
import importlib
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ClassInfo:
    """Information about a class for documentation."""

    name: str
    module_path: str
    docstring: str
    methods: List[Tuple[str, str]]  # (method_name, signature)
    attributes: List[Tuple[str, str]]  # (attr_name, description)
    bases: List[str]


@dataclass
class ModuleInfo:
    """Information about a module for documentation."""

    name: str
    path: str
    docstring: str
    classes: List[ClassInfo]
    functions: List[Tuple[str, str, str]]  # (name, signature, docstring)


class APIDocGenerator:
    """
    Generates API documentation in Markdown format from Python modules.

    Inspects Python packages and modules to extract documentation,
    class/function signatures, and docstrings, then formats them
    as structured Markdown suitable for MkDocs.

    Attributes:
        package_name: Full package name (e.g., 'pyadm1.components')
        output_dir: Directory to write documentation files
        exclude_private: Skip private members (starting with _)

    Example:

        >>> generator = APIDocGenerator("pyadm1.components")
        >>> generator.generate_all()
    """

    def __init__(self, package_name: str, output_dir: str = "docs/api_reference", exclude_private: bool = True):
        """
        Initialize documentation generator.

        Args:
            package_name: Package to document (e.g., 'pyadm1.components')
            output_dir: Output directory for markdown files
            exclude_private: Skip private members starting with _
        """
        self.package_name = package_name
        self.output_dir = Path(output_dir)
        self.exclude_private = exclude_private

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self) -> None:
        """
        Generate complete API documentation for the package.

        Creates a main overview file and separate files for each subpackage.
        """
        print(f"Generating API documentation for {self.package_name}...")

        # Import the package
        try:
            package = importlib.import_module(self.package_name)
        except ImportError as e:
            print(f"Error importing {self.package_name}: {e}")
            return

        # Get package structure
        subpackages = self._get_subpackages(package)

        # Generate main components.md file
        self._generate_main_file(package, subpackages)

        # Generate documentation for each subpackage
        for subpkg_name in subpackages:
            self._generate_subpackage_docs(subpkg_name)

        print(f"Documentation generated in {self.output_dir}")

    def _get_subpackages(self, package: Any) -> List[str]:
        """
        Get list of subpackages in a package.

        Args:
            package: Imported package object

        Returns:
            List of subpackage names
        """
        subpackages = []
        package_path = Path(package.__file__).parent

        for item in package_path.iterdir():
            if item.is_dir() and not item.name.startswith("_"):
                if (item / "__init__.py").exists():
                    subpackages.append(item.name)

        return sorted(subpackages)

    def _generate_main_file(self, package: Any, subpackages: List[str]) -> None:
        """
        Generate main components.md file with overview.

        Args:
            package: Imported package object
            subpackages: List of subpackage names
        """
        output_file = self.output_dir / "components.md"

        with open(output_file, "w", encoding="utf-8") as f:
            # Write header
            f.write("# Plant Components\n\n")

            # Write package docstring
            if package.__doc__:
                f.write(self._format_docstring(package.__doc__))
                f.write("\n\n")

            # Write subpackages overview
            f.write("## Subpackages\n\n")

            for subpkg_name in subpackages:
                subpkg_full = f"{self.package_name}.{subpkg_name}"
                try:
                    subpkg = importlib.import_module(subpkg_full)
                    brief_desc = self._get_brief_description(subpkg.__doc__)

                    f.write(f"### [{subpkg_name}]({subpkg_name}.md)\n\n")
                    if brief_desc:
                        f.write(f"{brief_desc}\n\n")
                except ImportError:
                    pass

            # Write base classes section
            f.write("## Base Classes\n\n")
            f.write("### Component Base\n\n")
            f.write("```python\n")
            f.write(f"from {self.package_name}.base import Component\n")
            f.write("```\n\n")

            # Document base.Component class
            self._document_class(f"{self.package_name}.base", "Component", f)

        print(f"Created {output_file}")

    def _generate_subpackage_docs(self, subpkg_name: str) -> None:
        """
        Generate documentation for a subpackage.

        Args:
            subpkg_name: Name of subpackage (e.g., 'biological')
        """
        subpkg_full = f"{self.package_name}.{subpkg_name}"
        output_file = self.output_dir / f"{subpkg_name}.md"

        try:
            subpkg = importlib.import_module(subpkg_full)
        except ImportError as e:
            print(f"Could not import {subpkg_full}: {e}")
            return

        with open(output_file, "w", encoding="utf-8") as f:
            # Write header
            title = self._format_title(subpkg_name)
            f.write(f"# {title} Components\n\n")

            # Write package docstring
            if subpkg.__doc__:
                f.write(self._format_docstring(subpkg.__doc__))
                f.write("\n\n")

            # Get all classes in subpackage
            classes = self._get_package_classes(subpkg_full)

            if classes:
                f.write("## Classes\n\n")

                for class_name in classes:
                    f.write(f"### {class_name}\n\n")
                    f.write("```python\n")
                    f.write(f"from {subpkg_full} import {class_name}\n")
                    f.write("```\n\n")

                    self._document_class(subpkg_full, class_name, f)
                    f.write("\n")

        print(f"Created {output_file}")

    def _document_class(self, module_path: str, class_name: str, file_handle) -> None:
        """
        Document a single class.

        Args:
            module_path: Full module path (e.g., 'pyadm1.components.biological')
            class_name: Name of class to document
            file_handle: Open file handle to write to
        """
        try:
            # Import module and get class
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            # Write class docstring
            if cls.__doc__:
                file_handle.write(self._format_docstring(cls.__doc__))
                file_handle.write("\n\n")

            # Write signature
            try:
                sig = inspect.signature(cls.__init__)
                params = []
                for param_name, param in sig.parameters.items():
                    if param_name != "self":
                        if param.default != inspect.Parameter.empty:
                            params.append(f"{param_name}={param.default!r}")
                        else:
                            params.append(param_name)

                if params:
                    file_handle.write("**Signature:**\n\n")
                    file_handle.write("```python\n")
                    file_handle.write(f"{class_name}(\n")
                    for i, param in enumerate(params):
                        comma = "," if i < len(params) - 1 else ""
                        file_handle.write(f"    {param}{comma}\n")
                    file_handle.write(")\n")
                    file_handle.write("```\n\n")
            except (ValueError, TypeError):
                pass

            # Document public methods
            methods = []
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                if self.exclude_private and name.startswith("_") and not name.startswith("__"):
                    continue
                if name.startswith("__") and name not in ["__init__"]:
                    continue
                methods.append((name, method))

            if methods:
                file_handle.write("**Methods:**\n\n")
                for method_name, method in methods:
                    if method_name == "__init__":
                        continue

                    file_handle.write(f"#### `{method_name}()`\n\n")

                    # Method signature
                    try:
                        sig = inspect.signature(method)
                        params = []
                        for param_name, param in sig.parameters.items():
                            if param_name != "self":
                                if param.default != inspect.Parameter.empty:
                                    params.append(f"{param_name}={param.default!r}")
                                else:
                                    params.append(param_name)

                        file_handle.write("```python\n")
                        file_handle.write(f"{method_name}({', '.join(params)})\n")
                        file_handle.write("```\n\n")
                    except (ValueError, TypeError):
                        pass

                    # Method docstring
                    if method.__doc__:
                        file_handle.write(self._format_docstring(method.__doc__))
                        file_handle.write("\n\n")

            # Document attributes from docstring
            self._document_attributes(cls, file_handle)

        except (ImportError, AttributeError) as e:
            file_handle.write(f"*Error documenting {class_name}: {e}*\n\n")

    def _document_attributes(self, cls: type, file_handle) -> None:
        """
        Extract and document class attributes from docstring.

        Args:
            cls: Class to document
            file_handle: Open file handle to write to
        """
        if not cls.__doc__:
            return

        # Look for Attributes section in docstring
        docstring = cls.__doc__
        if "Attributes:" not in docstring and "Attributes\n" not in docstring:
            return

        file_handle.write("**Attributes:**\n\n")

        # Parse attributes from docstring
        lines = docstring.split("\n")
        in_attributes = False

        for line in lines:
            stripped_line = line.strip()

            # Check if we're entering the Attributes section
            if stripped_line == "Attributes:" or stripped_line.startswith("Attributes:"):
                in_attributes = True
                continue

            if in_attributes:
                # Stop at next section (line that starts with capital letter and ends with :)
                if stripped_line and not line.startswith(" ") and re.match(r"^[A-Z][a-zA-Z\s]+:", stripped_line):
                    break

                # Extract attribute info (skip empty lines and separators)
                if stripped_line and not stripped_line.startswith("---"):
                    file_handle.write(f"- {stripped_line}\n")

        file_handle.write("\n")

    def _get_package_classes(self, package_path: str) -> List[str]:
        """
        Get all public classes defined in a package.

        Args:
            package_path: Full package path

        Returns:
            List of class names
        """
        try:
            package = importlib.import_module(package_path)

            # Get __all__ if defined
            if hasattr(package, "__all__"):
                classes = [name for name in package.__all__ if not name.startswith("_")]
            else:
                # Get all classes from module
                classes = []
                for name, obj in inspect.getmembers(package, inspect.isclass):
                    if not name.startswith("_"):
                        # Check if class is defined in this module
                        if obj.__module__.startswith(package_path):
                            classes.append(name)

            return sorted(classes)

        except (ImportError, AttributeError):
            return []

    def _format_docstring(self, docstring: str) -> str:
        """
        Format a docstring for markdown output.

        Intelligently handles code examples by detecting them and wrapping
        them in markdown code blocks. Supports both formats:
        - Example: (with blank line before code)
        - Example: >>> (code immediately after)

        Args:
            docstring: Raw docstring

        Returns:
            Formatted markdown string
        """
        if not docstring:
            return ""

        # Clean up indentation
        lines = docstring.split("\n")

        # Remove leading/trailing empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        # Find minimum indentation (excluding empty lines)
        min_indent = float("inf")
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)

        if min_indent == float("inf"):
            min_indent = 0

        # Remove common indentation
        cleaned_lines = []
        for line in lines:
            if line.strip():
                cleaned_lines.append(line[min_indent:])
            else:
                cleaned_lines.append("")

        # Process code examples and other sections with code blocks
        result_lines = []
        i = 0

        # Sections that might contain code blocks
        code_sections = [
            "Example",
            "Examples",
            "Returns",
            "Args",
            "Yields",
            "Raises",
            "Note",
            "Notes",
            "Warning",
            "Warnings",
            "See Also",
            "References",
            "Modules",
            "Classes",
            "Functions",
        ]

        # Create regex pattern for all code sections
        section_pattern = r"^(" + "|".join(code_sections) + r")[s]?:\s*$"

        print("cleaned_lines:", cleaned_lines)

        while i < len(cleaned_lines):
            line = cleaned_lines[i]

            # Check if this is a code section line
            if re.match(section_pattern, line.strip()):
                print("matched line:", line)
                result_lines.append(line)
                i += 1

                # Look ahead to see if code block follows
                if i < len(cleaned_lines):
                    # Check next few lines to determine if we have code
                    look_ahead_idx = i
                    has_blank_line = False

                    # Skip blank line if present
                    if look_ahead_idx < len(cleaned_lines) and not cleaned_lines[look_ahead_idx].strip():
                        has_blank_line = True
                        look_ahead_idx += 1

                    # Check if code block starts
                    has_code_block = False
                    if look_ahead_idx < len(cleaned_lines):
                        next_line = cleaned_lines[look_ahead_idx]
                        if next_line.strip().startswith(">>>"):
                            has_code_block = True

                    if has_code_block:
                        # Add blank line if not present
                        if has_blank_line:
                            result_lines.append(cleaned_lines[i])
                            i += 1
                        else:
                            result_lines.append("")

                        # Add code fence
                        result_lines.append("```python")

                        # Collect all code lines
                        while i < len(cleaned_lines):
                            code_line = cleaned_lines[i]
                            stripped = code_line.strip()

                            # Stop at empty line followed by non-code
                            if not stripped:
                                if i + 1 < len(cleaned_lines):
                                    next_stripped = cleaned_lines[i + 1].strip()
                                    if not next_stripped.startswith((">>>", "...")):
                                        break
                                result_lines.append(code_line)
                                i += 1
                                continue

                            # Stop at next section
                            if not stripped.startswith((">>>", "...")) and re.match(section_pattern, stripped):
                                break

                            # Stop at any new top-level section marker
                            if not stripped.startswith((">>>", "...")) and re.match(r"^[A-Z][a-z]+[s]?:", stripped):
                                break

                            result_lines.append(code_line)
                            i += 1

                        result_lines.append("```")
                        # continue here ensures we skip the normal line append at the end
                        continue

                    # No code block follows - process normally but don't re-add section header
                    # Continue processing from current position
                    continue

            # Normal line
            result_lines.append(line)
            i += 1

        print("result:", result_lines)

        return "\n".join(result_lines)

    def _format_title(self, name: str) -> str:
        """
        Format a package name as a title.

        Args:
            name: Package name (e.g., 'biological')

        Returns:
            Formatted title (e.g., 'Biological')
        """
        return name.replace("_", " ").title()

    def _get_brief_description(self, docstring: Optional[str]) -> str:
        """
        Extract brief description from docstring (first line).

        Args:
            docstring: Full docstring

        Returns:
            First non-empty line
        """
        if not docstring:
            return ""

        lines = docstring.strip().split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped:
                return stripped

        return ""


def generate_components_api_docs(output_dir: str = "docs/api_reference", package_name: str = "pyadm1.components") -> None:
    """
    Generate API documentation for PyADM1 components package.

    Creates comprehensive markdown documentation by inspecting the
    components package and all its subpackages, extracting docstrings,
    class signatures, and method documentation.

    Args:
        output_dir: Directory to write markdown files (default: 'docs/api_reference')
        package_name: Package to document (default: 'pyadm1.components')

    Example:

        >>> from pyadm1.utils.doc_generator import generate_components_api_docs
        >>>
        >>> # Generate with default settings
        >>> generate_components_api_docs()
        >>>
        >>> # Custom output directory
        >>> generate_components_api_docs(
        ...     output_dir="docs/custom_api",
        ...     package_name="pyadm1.components"
        ... )
    """
    generator = APIDocGenerator(package_name, output_dir)
    generator.generate_all()


if __name__ == "__main__":
    # Generate documentation when run as script
    generate_components_api_docs()
