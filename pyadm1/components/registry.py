# ============================================================================
# pyadm1/components/registry.py
# ============================================================================
"""
Component Registry for dynamic component loading and instantiation.

This module provides a registry system for managing and instantiating
biogas plant components dynamically.
"""

from typing import Dict, Type, Any, Optional
from .base import Component


class ComponentRegistry:
    """
    Registry for dynamically managing and creating component instances.

    The ComponentRegistry allows for plugin-like extensibility, enabling
    components to be registered and instantiated by name.

    Attributes:
        _registry (Dict[str, Type[Component]]): Internal registry mapping
            component names to their classes.

    Example:
        >>> registry = ComponentRegistry()
        >>> registry.register("Digester", Digester)
        >>> component = registry.create("Digester", "dig1", feedstock=feedstock)
    """

    def __init__(self):
        """Initialize an empty component registry."""
        self._registry: Dict[str, Type[Component]] = {}
        self._auto_register_components()

    def _auto_register_components(self) -> None:
        """
        Automatically register available components.

        This method attempts to import and register all standard components.
        If imports fail (e.g., missing dependencies), those components are skipped.
        """
        # Try to register biological components
        try:
            from pyadm1.components.biological.digester import Digester

            self._registry["Digester"] = Digester
        except ImportError:
            pass

        try:
            from pyadm1.components.biological.hydrolysis import Hydrolysis

            self._registry["Hydrolysis"] = Hydrolysis
        except ImportError:
            pass

        try:
            from pyadm1.components.biological.separator import Separator

            self._registry["Separator"] = Separator
        except ImportError:
            pass

        # Try to register energy components
        try:
            from pyadm1.components.energy.chp import CHP

            self._registry["CHP"] = CHP
        except ImportError:
            pass

        try:
            from pyadm1.components.energy.heating import HeatingSystem

            self._registry["HeatingSystem"] = HeatingSystem
        except ImportError:
            pass

        try:
            from pyadm1.components.energy.boiler import Boiler

            self._registry["Boiler"] = Boiler
        except ImportError:
            pass

        try:
            from pyadm1.components.energy.gas_storage import GasStorage

            self._registry["GasStorage"] = GasStorage
        except ImportError:
            pass

        try:
            from pyadm1.components.energy.flare import Flare

            self._registry["Flare"] = Flare
        except ImportError:
            pass

    def register(self, name: str, component_class: Type[Component]) -> None:
        """
        Register a component class with a given name.

        Args:
            name (str): Name to register the component under.
            component_class (Type[Component]): Component class to register.

        Raises:
            ValueError: If name is already registered.

        Example:
            >>> registry.register("CustomDigester", CustomDigester)
        """
        if name in self._registry:
            raise ValueError(f"Component '{name}' is already registered")
        self._registry[name] = component_class

    def unregister(self, name: str) -> None:
        """
        Unregister a component class.

        Args:
            name (str): Name of the component to unregister.

        Raises:
            KeyError: If component name is not registered.
        """
        if name not in self._registry:
            raise KeyError(f"Component '{name}' is not registered")
        del self._registry[name]

    def create(self, name: str, component_id: str, **kwargs: Any) -> Component:
        """
        Create a component instance by name.

        Args:
            name (str): Name of the registered component class.
            component_id (str): Unique identifier for the component instance.
            **kwargs: Additional keyword arguments passed to component constructor.

        Returns:
            Component: Instantiated component.

        Raises:
            KeyError: If component name is not registered.

        Example:
            >>> component = registry.create("Digester", "dig1",
            ...                            feedstock=feedstock, V_liq=2000)
        """
        if name not in self._registry:
            raise KeyError(f"Component '{name}' is not registered. " f"Available components: {list(self._registry.keys())}")

        component_class = self._registry[name]
        return component_class(component_id=component_id, **kwargs)

    def get_registered_components(self) -> Dict[str, Type[Component]]:
        """
        Get all registered component classes.

        Returns:
            Dict[str, Type[Component]]: Dictionary mapping names to component classes.
        """
        return self._registry.copy()

    def is_registered(self, name: str) -> bool:
        """
        Check if a component name is registered.

        Args:
            name (str): Component name to check.

        Returns:
            bool: True if registered, False otherwise.
        """
        return name in self._registry

    def list_components(self) -> list[str]:
        """
        Get a list of all registered component names.

        Returns:
            list[str]: List of registered component names.
        """
        return list(self._registry.keys())


# Global registry instance
_global_registry: Optional[ComponentRegistry] = None


def get_registry() -> ComponentRegistry:
    """
    Get the global component registry instance.

    Returns:
        ComponentRegistry: The global registry instance.

    Example:
        >>> registry = get_registry()
        >>> component = registry.create("Digester", "dig1", feedstock=feedstock)
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ComponentRegistry()
    return _global_registry


def register_component(name: str, component_class: Type[Component]) -> None:
    """
    Register a component in the global registry.

    Args:
        name (str): Name to register the component under.
        component_class (Type[Component]): Component class to register.

    Example:
        >>> register_component("MyCustomComponent", MyCustomComponent)
    """
    registry = get_registry()
    registry.register(name, component_class)
