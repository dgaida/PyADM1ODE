# Substrates

A substrate in PyADM1ODE is a file of laboratory values: dry matter, feed
analysis, methane potential and the state in which the material reaches the
digester. From these values the model computes the digester influent – how
much degradable material arrives, how fast it degrades, and how much nitrogen
and buffer it brings along.

```python
from pyadm1 import Feedstock

# Two substrates of the library, chosen by file name without extension
feedstock = Feedstock(["maize_silage_milk_ripeness", "cattle_manure"])
```

The files live under
[`data/substrates/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates).
The amount of each substrate (`Q_substrates`, in tonnes of fresh matter per
day) is set at the digester, in the order of the list above.

## What this section answers

| Question | Page |
| --- | --- |
| Which example substrates exist, and which one suits what? | [Example substrates](library.md) |
| What do the values of a substrate file mean, and which ranges are realistic? | [Parameters & ranges](parameters.md) |
| What changes in the simulation when a value is different? | [Influence of the values](influence.md) |
| Where do the values of the bundled files come from? | [Origin of the values](sources.md) |

!!! note "Reference values – define your own substrates for real plants"
    The bundled substrates describe **typical material**, not the substrate of
    a particular plant. They are suitable for demos, tests and studies, **but
    are no substitute for measured data**. To model a real plant, characterise
    your own substrate (TS / VS, Weender and cell-wall analysis, BMP from a
    batch test) and add a new file under `data/substrates/`. The page
    [Parameters & ranges](parameters.md) shows which values that takes.

## Your own substrate in three steps

1. Copy the library file of the most similar material, for example  
   `maize_silage_milk_ripeness.yaml` to `my_maize.yaml`.  
2. Enter your own laboratory values. Required are `TS`, `VS`, `RP`, `RL`, `pH`  
   and one of the three values `BMP`, `D_VS` or `aXI`; everything else is optional.  
3. Load the file by its name or its path:  

```python
from pyadm1 import Feedstock
from pyadm1.substrates import load_substrate

feedstock = Feedstock(["my_maize", "cattle_manure"])       # lives in data/substrates/
my_maize = load_substrate("path/to/my_maize.yaml")          # or by path
```

A misspelt parameter name raises an error; it is not silently skipped.

How the file becomes the 38-column ADM1 influent stream is described on the
page [ADM1 implementation](../user_guide/adm1_implementation.md).
