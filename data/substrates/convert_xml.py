import xml.etree.ElementTree as ET
import pandas as pd

def xml_to_table(xml_path: str) -> pd.DataFrame:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []

    # Iterate over each substrate
    for sub in root.findall(".//substrate"):
        substrate_id = sub.get("id", "")
        substrate_name = (sub.findtext("name") or "").strip()
        substrate_class = (sub.findtext("substrate_class") or "").strip()

        # We treat each direct child section (e.g., Weender, Phys, AD) as a "section"
        for section in list(sub):
            section_tag = section.tag

            # Skip simple non-section elements
            if section_tag in {"name", "substrate_class"}:
                continue

            # Find physValue nodes within this section (also works if nested)
            for pv in section.findall(".//physValue"):
                symbol = pv.get("symbol", "").strip()
                value = (pv.findtext("value") or "").strip()
                unit = (pv.findtext("unit") or "").strip()
                label = (pv.findtext("label") or "").strip()

                # Collect reference text (including multi-line content)
                ref_elem = pv.find("reference")
                reference = ""
                if ref_elem is not None:
                    reference = "".join(ref_elem.itertext()).strip()

                rows.append({
                    "substrate_id": substrate_id,
                    "substrate_name": substrate_name,
                    "substrate_class": substrate_class,
                    "section": section_tag,
                    "symbol": symbol,
                    "value": value,
                    "unit": unit,
                    "label": label,
                    "reference": reference,
                })

    df = pd.DataFrame(rows)

    # Optional: try to convert numeric values
    # This keeps non-numeric strings as-is.
    def to_float_maybe(x: str):
        x2 = x.replace(",", ".")
        try:
            return float(x2)
        except Exception:
            return x

    if not df.empty and "value" in df.columns:
        df["value"] = df["value"].apply(to_float_maybe)

    return df


if __name__ == "__main__":
    xml_file = "substrate_gummersbach - Kopie.xml"  # <-- change this
    df = xml_to_table(xml_file)

    # CSV
    # df.to_csv("substrates.csv", index=False, encoding="utf-8-sig")

    # Excel (requires openpyxl installed, which is common)
    df.to_excel("substrates.xlsx", index=False)

    print("Wrote substrates.xlsx")
