# Dataset Structure

The dataset is organised like a **workbook**: there are several plants, and for each
plant several task variants. This page explains how everything fits together.

## Plants as building blocks

Each plant lives in its own folder. There are currently eleven example plants that
differ in size and equipment:

| Plant    | Short description                                                        |
| -------- | ----------------------------------------------------------------------- |
| **BGA1** | Large plant: two fermenters, secondary digester, digestate store, biogas upgrading, separator |
| **BGA2** | Small plant: one fermenter, secondary digester, digestate store, combined heat and power unit |
| **BGA3** | Medium plant: two fermenters, secondary digester, digestate store, combined heat and power unit |
| **BGA4** | Real plant (250 kW): digester, post-digester, digestate store, combined heat and power unit |
| **BGA5** | Small slurry plant (75 kW): fermenter and gas-tight digestate store only, no secondary digester |
| **BGA6** | Energy-crop plant (360 kW): horizontal plug-flow digester (thermophilic), secondary digester, digestate store, separator with press-water recirculation |
| **BGA7** | Flexibilised plant (800 kW): two fermenters in series, gas-tight and open digestate store, separator, **two** CHP units |
| **BGA8** | Four-stage chain (300 kW): fermenter, **two identical secondary digesters in series**, digestate store, combined heat and power unit |
| **BGA9** | Four-stage chain (400 kW): fermenter, secondary digester, **two gas-tight digestate stores in series**, combined heat and power unit |
| **BGA10** | **Two parallel lines** (750 kW): one fermenter and one secondary digester per line, shared digestate store, combined heat and power unit — five tanks |
| **BGA11** | **Two gas consumers** (250 kW + 350 m³/h): two fermenters, secondary digester, digestate store, combined heat and power unit **and** biogas upgrading in parallel |

!!! info "BGA = biogas plant"
    "BGA" simply stands for the German *Biogasanlage* (biogas plant). The number
    distinguishes the eleven examples.

## Variants: the same plant, described differently

For each plant there is the **same** biogas plant, but **described in different
ways**. This makes it possible to test whether the AI is robust – regardless of
whether the description is long, short, in English or a sketch.

Two properties are combined here:

**1. The form of the description**

- **detailed text** – an explanatory prose description  
- **terse text** – only the key figures  
- **English text** – the same plant in English  
- **sketch** – a drawing of the plant (image)  

**2. How complete the description is**

- **complete** (suffix `_full`): All required information is in the description. The  
  AI does not need to ask anything.  
- **incomplete**: Some information is missing (e.g. the operating temperature). The  
  AI has to **ask** the oracle for it — a guess is practically never close enough.

!!! note "For the sketch the text adds only what is missing"
    In the complete variant `…_sketch_full` the supplementary text states **only
    what the image does not carry itself**. The sketch labels dimensions and CHP
    power — the supplement does not repeat that, it supplies the operating
    temperature, the fill level, the gas space, the efficiencies and the path of
    the digestate. That keeps the task tied to actually reading the sketch.

!!! example "Example"
    `BGA2_terse_de_full` means: plant **BGA2**, **terse** description, in **German**
    (`de`), with **all** information (`full`).

## The reference solution ("Gold")

Each plant comes with a **reference solution** – similar to the answer sheet for a
school exercise. It describes the correctly built plant and serves as the benchmark
against which the AI's result is measured. All variants of a plant share the same
reference solution, because it is always the same plant.

## What the folders look like

In simplified form, the storage layout looks like this:

```text
Dataset/
  BGA1/                     ← plant 1 (one folder per plant)
    BGA1_text_de.json         detailed text (German), incomplete
    BGA1_text_de_full.json    detailed text (German), complete
    BGA1_text_en.json         English text
    BGA1_terse_de.json        terse description
    BGA1_sketch.json          sketch only
    BGA1_sketch.png           the sketch image
    gold.py                   the shared reference solution
  BGA2/  …                   ← plant 2 (same layout)
  BGA3/  …                   ← plant 3 (same layout)
  BGA4/  …                   ← plant 4 (same layout)
  BGA5/  …                   ← plant 5 (same layout, no secondary digester)
  BGA6/  …                   ← plant 6 (same layout, plug-flow digester)
  BGA7/  …                   ← plant 7 (same layout, two CHP units)
  BGA8/  …                   ← plant 8 (same layout, serial chain)
  BGA9/  …                   ← plant 9 (same layout, two stores)
  BGA10/ …                   ← plant 10 (same layout, two parallel lines)
  BGA11/ …                   ← plant 11 (same layout, CHP + upgrading)
```

!!! note "What is a `.json` file?"
    A `.json` file is a **text file in a fixed format** that a computer can read
    easily. You can think of it as a filled-in form with clearly named fields. What
    exactly is in such a form is explained on the page
    [A Data Point in Detail](datenpunkt.md).

Each of these task files is what we call a **data point**. What is inside one is what
we look at next.
