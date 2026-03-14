# 🏓 BallSatzSieg – TT Performance Rating

**Schätzung der tatsächlichen Spielstärke aus Satzergebnissen via Maximum-Likelihood**

→ **[App öffnen](https://tt-performance-rating.streamlit.app)**

---

## Was ist das?

Das offizielle TTR-Rating misst langfristige Spielstärke und reagiert nur auf Sieg oder Niederlage. Diese App berechnet ein **Performance Rating** direkt aus den Satzergebnissen – unabhängig davon ob man gewonnen oder verloren hat.

Ein knappes 2:3 gegen einen deutlich stärkeren Gegner kann ein höheres Performance Rating ergeben als ein klares 3:0 gegen einen schwachen Gegner.

## Features

- **Web-Import** – direkt von bettv.tischtennislive.de: eine URL eingeben, alle Spiele einer Saison werden automatisch geladen
- **Saisonverlauf** – Performance Rating pro Punktspiel mit Fehlerbalken, Bootstrap-Dichte, drei optionale Overlays:
  - 📈 Linearer Trend (gewichtet, mit Konfidenzbändern)
  - 🔄 Gleitender Verlauf (rollierende 5 Punktspiele, mit 1σ/2σ-Band)
  - 🏅 LivePZ-Referenz (offizieller Wert zum Vergleich)
- **Subset-Analysen** – Vorrunde, Rückrunde, Heim, Auswärts, letzte 5 Punktspiele
- **Bootstrap-Konfidenzintervall** – parametrisch auf Rally-Ebene, 1σ und 2σ
- **Modelldiagnose** – χ²-Test auf Konstanz der Spielstärke innerhalb einer Partie
- **Manuelle Eingabe** – alternativ zu Web-Import, auch für Turniere

## Methodik

Das Modell basiert auf einer einzigen Annahme: jeder Punkt wird mit einer festen Wahrscheinlichkeit *p* gewonnen. Daraus folgt über eine analytische Kaskade:

```
p (Rally)  →  p_Satz  →  p_Match  →  TTR*
```

Die Rally-Wahrscheinlichkeit wird per Maximum-Likelihood aus den Satzergebnissen geschätzt (äquivalent zur einfachen Punktquote). Das Performance Rating ergibt sich durch Inversion der TTR-Formel.

Eine ausführliche Beschreibung der Methodik findet sich in [PAPER.md](PAPER.md).

## Installation

```bash
pip install streamlit scipy pandas numpy requests beautifulsoup4 plotly
streamlit run BallSatzSieg.py
```

## Abhängigkeiten

| Paket | Zweck |
|-------|-------|
| `streamlit` | Web-UI |
| `scipy` | MLE-Optimierung, Bootstrap, KDE |
| `numpy` | Numerik |
| `pandas` | Datentabellen |
| `plotly` | Interaktive Diagramme |
| `requests` | Web-Import |
| `beautifulsoup4` | HTML-Parser für bettv |

## Lizenz

Privates Projekt – kein öffentlicher Einsatz ohne Rücksprache.

---

*Torsten Langer · 2026*
