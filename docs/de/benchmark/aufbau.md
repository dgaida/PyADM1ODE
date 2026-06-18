# Aufbau des Datensatzes

Der Datensatz ist wie ein **Aufgabenheft** aufgebaut: Es gibt mehrere Anlagen, und
zu jeder Anlage mehrere Aufgaben-Varianten.

## Anlagen als Bausteine

Jede Anlage liegt in einem eigenen Ordner. Das sind zum Beispiel die ersten 3 Beispielanlagen:

| Anlage   | Kurzbeschreibung                                                        |
| -------- | ----------------------------------------------------------------------- |
| **BGA1** | Große Anlage: zwei Fermenter, Nachgärer, Gärrestlager, Biogasaufbereitung, Separator |
| **BGA2** | Kleine Anlage: ein Fermenter, Nachgärer, Gärrestlager, Blockheizkraftwerk |
| **BGA3** | Mittlere Anlage: zwei Fermenter, Nachgärer, Gärrestlager, Blockheizkraftwerk |

!!! info "BGA = Biogasanlage"
    „BGA" steht für **B**io**g**as**a**nlage. Die Nummer unterscheidet die
    drei Beispiele.

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

**2. Die Vollständigkeit der Angaben**

- **vollständig**: Alle nötigen Angaben stehen in der Beschreibung.  
  Die KI muss nichts nachfragen.  
- **unvollständig**: Es fehlen Angaben (z. B. die Betriebstemperatur). Die KI muss  
  diese **erfragen** oder sinnvoll **ergänzen**.

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
```

!!! note "Was ist eine `.json`-Datei?"
    Eine `.json`-Datei ist eine **Textdatei in einem festen Format**, die der
    Computer leicht lesen kann. Man kann sie sich wie ein ausgefülltes Formular mit
    klar benannten Feldern vorstellen. Was genau in so einem Formular steht, erklärt
    die Seite [Ein Datenpunkt im Detail](datenpunkt.md).
