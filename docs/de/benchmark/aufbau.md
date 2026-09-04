# Aufbau des Datensatzes

Der Datensatz ist wie ein **Aufgabenheft** aufgebaut: Es gibt mehrere Anlagen, und
zu jeder Anlage mehrere Aufgaben-Varianten.

## Anlagen als Bausteine

Jede Anlage liegt in einem eigenen Ordner. Das sind die elf Beispielanlagen:

| Anlage   | Kurzbeschreibung                                                        |
| -------- | ----------------------------------------------------------------------- |
| **BGA1** | Große Anlage: zwei Fermenter, Nachgärer, Gärrestlager, Biogasaufbereitung, Separator |
| **BGA2** | Kleine Anlage: ein Fermenter, Nachgärer, Gärrestlager, Blockheizkraftwerk |
| **BGA3** | Mittlere Anlage: zwei Fermenter, Nachgärer, Gärrestlager, Blockheizkraftwerk |
| **BGA4** | Reale Anlage (250 kW): Fermenter, Nachgärer, Gärproduktlager, Blockheizkraftwerk |
| **BGA5** | Güllekleinanlage (75 kW): nur Fermenter und gasdicht abgedecktes Gärrestlager, kein Nachgärer |
| **BGA6** | NawaRo-Anlage (360 kW): liegender Pfropfenstromfermenter (thermophil), Nachgärer, Gärrestlager, Separator mit Presswasser-Rezirkulation |
| **BGA7** | Flexibilisierte Anlage (800 kW): zwei Fermenter in Reihe, gasdichtes und offenes Gärrestlager, Separator, **zwei** Blockheizkraftwerke |
| **BGA8** | Anlage mit vierstufiger Kette (300 kW): Fermenter, **zwei baugleiche Nachgärer in Reihe**, Gärrestlager, Blockheizkraftwerk |
| **BGA9** | Anlage mit vierstufiger Kette (400 kW): Fermenter, Nachgärer, **zwei gasdichte Gärrestlager in Reihe**, Blockheizkraftwerk |
| **BGA10** | Anlage mit **zwei parallelen Linien** (750 kW): je Linie ein Fermenter und ein Nachgärer, gemeinsames Gärrestlager, Blockheizkraftwerk — fünf Behälter |
| **BGA11** | Anlage mit **zwei Gasabnehmern** (250 kW + 350 m³/h): zwei Fermenter, Nachgärer, Gärrestlager, Blockheizkraftwerk **und** Biogasaufbereitung parallel |

!!! info "BGA = Biogasanlage"
    „BGA" steht für **B**io**g**as**a**nlage. Die Nummer unterscheidet die
    elf Beispiele.

## Varianten

Pro Anlage gibt es **unterschiedliche Beschreibungen**, welche diselbe Biogasanlage
beschreiben. So lässt sich prüfen, ob die KI robust ist egal ob die Beschreibung lang,
kurz, auf Englisch oder als Skizze vorliegt.

Zwei Eigenschaften werden dabei kombiniert:

**1. Die Form der Beschreibung**

- **ausführlicher Text** – ein erklärender Fließtext  
- **knapper Text** (terse) – nur die wichtigsten Eckdaten  
- **englischer Text** – dieselbe Anlage auf Englisch  
- **Skizze** – eine Zeichnung der Anlage (Bild)  

**2. Wie vollständig die Beschreibung ist**

- **vollständig**: Alle nötigen Angaben stehen in der Beschreibung.  
  Die KI muss nichts nachfragen.  
- **unvollständig**: Es fehlen Angaben (z. B. die Betriebstemperatur). Die KI muss  
  diese beim Oracle **erfragen** — geraten wird der Wert praktisch nie genau genug.

!!! note "Bei der Skizze ergänzt der Text nur das Fehlende"
    In der vollständigen Variante `…_sketch_full` steht im Zusatztext **nur, was
    das Bild nicht selbst hergibt**. Die Skizze beschriftet Maße und
    BHKW-Leistung — der Zusatztext wiederholt das nicht, sondern liefert
    Betriebstemperatur, Füllgrad, Gasraum, Wirkungsgrade und den Weg des
    Gärrests. So bleibt die Aufgabe daran gebunden, die Skizze tatsächlich zu
    lesen.

## Die Musterlösung („Gold")

Zu jeder Anlage gehört eine **Musterlösung**. Sie beschreibt die korrekt aufgebaute Anlage und
dient als Maßstab, an dem das Ergebnis der KI gemessen wird. Alle Varianten einer Anlage
teilen sich dieselbe Musterlösung, weil es ja immer dieselbe Anlage ist.

## Wie die Ordner aussehen

Vereinfacht sieht die Ablage so aus:

```text
Datensatz/
  BGA1/                     ← Anlage 1 (ein Ordner pro Anlage)
    BGA1_text_de.json         ausführlicher Text (Deutsch), unvollständig
    BGA1_text_de_full.json    ausführlicher Text (Deutsch), vollständig
    BGA1_text_en.json         englischer Text
    BGA1_terse_de.json        knappe Beschreibung
    BGA1_sketch.json          nur Skizze
    BGA1_sketch.png           das Skizzen-Bild
    gold.py                   die gemeinsame Musterlösung
  BGA2/  …                   ← Anlage 2 (gleicher Aufbau)
  BGA3/  …                   ← Anlage 3 (gleicher Aufbau)
  BGA4/  …                   ← Anlage 4 (gleicher Aufbau)
  BGA5/  …                   ← Anlage 5 (gleicher Aufbau, ohne Nachgärer)
  BGA6/  …                   ← Anlage 6 (gleicher Aufbau, Pfropfenstrom)
  BGA7/  …                   ← Anlage 7 (gleicher Aufbau, zwei BHKW)
  BGA8/  …                   ← Anlage 8 (gleicher Aufbau, serielle Kette)
  BGA9/  …                   ← Anlage 9 (gleicher Aufbau, zwei Lager)
  BGA10/ …                   ← Anlage 10 (gleicher Aufbau, zwei Linien)
  BGA11/ …                   ← Anlage 11 (gleicher Aufbau, BHKW + Aufbereitung)
```

!!! note "Was ist eine `.json`-Datei?"
    Eine `.json`-Datei ist eine **Textdatei in einem festen Format**, die der
    Computer leicht lesen kann. Man kann sie sich wie ein ausgefülltes Formular mit
    klar benannten Feldern vorstellen. Was genau in so einem Formular steht, erklärt
    die Seite [Ein Datenpunkt im Detail](datenpunkt.md).
