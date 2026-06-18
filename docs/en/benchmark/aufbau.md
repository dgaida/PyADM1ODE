# Dataset Structure

The dataset is organised like a **workbook**: there are several plants, and for each
plant several task variants. This page explains how everything fits together.

## Plants as building blocks

Each plant lives in its own folder. There are currently three example plants that
differ in size and equipment:

| Plant    | Short description                                                        |
| -------- | ----------------------------------------------------------------------- |
| **BGA1** | Large plant: two fermenters, secondary digester, digestate store, biogas upgrading, separator |
| **BGA2** | Small plant: one fermenter, secondary digester, digestate store, combined heat and power unit |
| **BGA3** | Medium plant: two fermenters, secondary digester, digestate store, combined heat and power unit |

!!! info "BGA = biogas plant"
    "BGA" simply stands for the German *Biogasanlage* (biogas plant). The number
    distinguishes the three examples.

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

**2. The completeness of the information**

- **complete** (suffix `_full`): All required information is in the description. The
  AI does not need to ask anything.
- **incomplete**: Some information is missing (e.g. the operating temperature). The
  AI has to **ask** for it or fill it in sensibly.

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
```

!!! note "What is a `.json` file?"
    A `.json` file is a **text file in a fixed format** that a computer can read
    easily. You can think of it as a filled-in form with clearly named fields. What
    exactly is in such a form is explained on the page
    [A Data Point in Detail](datenpunkt.md).

Each of these task files is what we call a **data point**. What is inside one is what
we look at next.
