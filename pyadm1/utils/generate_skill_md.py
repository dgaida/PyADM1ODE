import inspect
import os
from pyadm1.substrates.feedstock import Feedstock
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.configurator.plant_configurator import PlantConfigurator
from pyadm1.components.mechanical.pump import Pump
from pyadm1.components.mechanical.mixer import Mixer
from pyadm1.components.feeding.substrate_storage import SubstrateStorage
from pyadm1.components.feeding.feeder import Feeder

def get_full_doc(obj):
    doc = inspect.getdoc(obj)
    if not doc:
        return "No documentation available."
    return doc

def generate_skill_md(output_path):
    classes_to_document = [
        Feedstock,
        BiogasPlant,
        PlantConfigurator,
        Pump,
        Mixer,
        SubstrateStorage,
        Feeder
    ]

    content = "# PyADM1ODE Simulation Model Creation Skill\n\n"
    content += "This document provides the full API documentation for the classes and methods required to build a PyADM1ODE biogas plant simulation model.\n\n"

    for cls in classes_to_document:
        content += f"## {cls.__name__}\n\n"
        content += f"```python\n{cls.__name__}\n```\n\n"
        content += f"{get_full_doc(cls)}\n\n"

        methods = [m for m in inspect.getmembers(cls, predicate=inspect.isroutine)
                   if not m[0].startswith('_') or m[0] == '__init__']

        if methods:
            content += f"### Methods for {cls.__name__}\n\n"
            for name, func in methods:
                try:
                    sig = str(inspect.signature(func))
                except (ValueError, TypeError):
                    sig = "(...)"
                content += f"#### {name}\n\n"
                content += f"```python\n{name}{sig}\n```\n\n"
                content += f"{get_full_doc(func)}\n\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Generated Skill.md at: {output_path}")

if __name__ == "__main__":
    generate_skill_md('docs/en/user_guide/Skill.md')
    generate_skill_md('docs/de/user_guide/Skill.md')
