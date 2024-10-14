#    Copyright 2024 SRI Lab @ ETH Zurich, LatticeFlow AI, INSAIT
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Callable, Optional


class ComponentRegistry:
    """
    A class that represents a component registry.

    The ComponentRegistry class is responsible for registering and managing logic and configuration classes
    for different components. It provides methods to register, retrieve, and manage these classes.

    Attributes:
        _registered_logic_classes (dict[str, type]): A dictionary that stores the registered logic classes.
        _registered_config_classes (dict[str, type]): A dictionary that stores the registered configuration classes.
        _categories (dict[str, str]): A dictionary that stores the categories of the registered classes.
        _logic_base_cls (type): The base class for logic classes.
        _config_base_cls (type): The base class for configuration classes.
    """

    def __init__(self, logic_cls: type, config_cls: type, category: str = "default"):
        self._registered_logic_classes: dict[str, type] = {}
        self._registered_config_classes: dict[str, type] = {}
        self._categories: dict[str, str] = {}
        self._logic_base_cls = logic_cls
        self._config_base_cls = config_cls

    def get_category(self, name: str) -> str:
        """
        Get the category of a registered class.

        Args:
            name (str): The name of the class.

        Returns:
            str: The category of the class.

        Raises:
            ValueError: If the class name is not found in the categories dictionary.
        """
        if name not in self._categories:
            raise ValueError(f"{name} config cls not found")
        return self._categories[name]

    def get_logic_base_cls(self) -> type:
        """
        Get the base class for logic classes.
        This value is static and doesn't change after initialization.

        Returns:
            type: The base class for logic classes.
        """
        return self._logic_base_cls

    def get_config_base_cls(self) -> type:
        """
        Get the base class for configuration classes.
        This value is static and doesn't change after initialization.

        Returns:
            type: The base class for configuration classes.
        """
        return self._config_base_cls

    def register_logic_config_classes(self, name, logic_cls, config_cls, category: str = "default"):
        """
        Register logic and configuration classes for a component.

        Args:
            name (str): The name of the component.
            logic_cls (type): The logic class to register.
            config_cls (type): The configuration class to register.
            category (str, optional): The category of the component. Defaults to "default".

        Raises:
            AssertionError: If the logic class or configuration class is not a subclass of the base classes.
            ValueError: If the logic class or configuration class is already registered.
        """
        assert issubclass(logic_cls, self._logic_base_cls)
        assert issubclass(config_cls, self._config_base_cls)

        self.register_logic_cls(name, logic_cls)
        self.register_config_cls(name, config_cls)

        self._categories[name] = category

    def register_logic_cls(self, name: str, entry: type):
        """
        Register a logic class for a component.

        Args:
            name (str): The name of the component.
            entry (type): The logic class to register.

        Raises:
            ValueError: If the logic class is already registered.
        """
        if name in self._registered_logic_classes:
            raise ValueError(f"{name} logic cls is already registered")

        self._registered_logic_classes[name] = entry

    def register_config_cls(self, name: str, entry: type):
        """
        Register a configuration class for a component.

        Args:
            name (str): The name of the component.
            entry (type): The configuration class to register.

        Raises:
            ValueError: If the configuration class is already registered.
        """
        if name in self._registered_config_classes:
            raise ValueError(f"{name} config cls is already registered")
        self._registered_config_classes[name] = entry

    def get_config_cls(self, name: str) -> type:
        """
        Get the configuration class for a component.

        Args:
            name (str): The name of the component.

        Returns:
            type: The configuration class.

        Raises:
            ValueError: If the configuration class is not found.
        """
        if name not in self._registered_config_classes:
            raise ValueError(f"{name} config cls not found")
        return self._registered_config_classes[name]

    def get_logic_cls(self, name: str) -> type:
        """
        Get the logic class for a component.

        Args:
            name (str): The name of the component.

        Returns:
            type: The logic class.

        Raises:
            ValueError: If the logic class is not found.
        """
        if name not in self._registered_logic_classes:
            raise ValueError(f"{name} logic cls not found")
        return self._registered_logic_classes[name]


class Registry(object):
    """
    A class that represents a global registry for the different registry components.


    The `Registry` class allows registering and retrieving registry components by name.

    Attributes:
        _registered (dict[str, ComponentRegistry]): A dictionary that stores the registered components.

    Methods:
        register(name: str, entry: ComponentRegistry): Registers a component with the given name.
        get(name: str) -> ComponentRegistry: Retrieves a registered component by name.
    """

    def __init__(self):
        self._registered: dict[str, ComponentRegistry] = {}

    def register(self, name: str, entry: ComponentRegistry):
        """
        Registers a component with the given name.

        Args:
            name (str): The name of the component registry.
            entry (ComponentRegistry): The component registry to be registered.

        Raises:
            ValueError: If a component with the same name is already registered.
        """
        if name in self._registered:
            raise ValueError(f"{name} is already registered")
        self._registered[name] = entry

    def get(self, name: str) -> ComponentRegistry:
        """
        Retrieves a registered component by name.

        Args:
            name (str): The name of the component registry to retrieve.

        Returns:
            ComponentRegistry: The component registry corresponding to the ```name``` type of classes.

        Raises:
            ValueError: If a component with the given name is not found.
        """
        if name not in self._registered:
            raise ValueError(f"{name} not found")
        return self._registered[name]


registry = Registry()
BENCHMARK_PROCESSORS: dict[str, Optional[Callable]] = {}
