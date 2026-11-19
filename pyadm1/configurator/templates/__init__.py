"""
Plant configuration templates.

This package is a placeholder: no template classes are implemented yet. The
ready-made layouts live on :class:`pyadm1.configurator.PlantConfigurator`.

Example:
    >>> from pyadm1 import BiogasPlant, Feedstock
    >>> from pyadm1.configurator import PlantConfigurator
    >>>
    >>> feedstock = Feedstock(["maize_silage_milk_ripeness", "cattle_manure"])
    >>> configurator = PlantConfigurator(BiogasPlant("Two-stage plant"), feedstock)
    >>> components = configurator.create_two_stage_plant()
"""
