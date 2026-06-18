# Glossary

The most important terms around the benchmark dataset – briefly explained and
without jargon.

## Plant components

**Fermenter**
: The main vessel in which bacteria turn slurry and other materials into biogas.
Usually heated (often around 40 °C).

**Secondary digester**
: A downstream vessel that further digests the not-yet-fully-fermented remainder,
gaining additional biogas.

**Digestate store**
: A storage vessel for the fully digested remainder (the "digestate"). Often
unheated.

**Combined heat and power unit (CHP / BHKW)**
: An engine that burns the biogas and produces **electricity and heat** from it.

**Biogas upgrading (BGAA)**
: A unit that cleans the biogas until it can be fed into the natural gas grid as
**biomethane**.

**Separator**
: A device that splits the digestate into a **solid** and a **liquid** part.

**Gas storage / dome roof**
: The space where the produced biogas is buffered – often a dome-shaped roof on top
of the vessel.

**Emergency flare**
: A safety device that burns off surplus biogas in a controlled way when it cannot
currently be used.

## Key figures

**Fill level**
: The share of the vessel filled with liquid (e.g. 90 %). The rest stays free as gas
space.

**V_liq (liquid volume)**
: The volume of the liquid in the vessel, in cubic metres (m³).

**V_gas (gas space)**
: The volume of the gas space above the liquid – that is, **dome roof plus unfilled
headspace**.

**Operating temperature**
: The temperature at which a vessel is operated. "Mesophilic" means around 40 °C.

## Benchmark terms

**Dataset**
: The entire collection of tasks (all plants and their variants).

**Data point**
: A **single task** – a particular plant in a particular form of description.

**Variant**
: The same plant, described differently (detailed, terse, English, as a sketch;
complete or incomplete).

**Regime**
: Indicates whether a task is **complete** (all information present) or
**incomplete** (information missing).

**Reference**
: The **correct plant** for a task, against which the AI's result is measured.

**Reference solution ("Gold")**
: A known-correct implementation of the plant. It serves as a benchmark and to
validate the scoring procedure.

**Oracle**
: An "expert" that the AI may query for **missing information**. It provides the
correct values and also recognises different phrasings of the same question.

**Three-part score**
: Each result receives three percentages – for **structure**, **measures** and the
handling of **gaps**. More on this on the [Scoring & Workflow](bewertung.md) page.

**Substrates**
: The input materials used (e.g. maize silage, cattle slurry). In the benchmark they
are part of the description but are **not** scored.
