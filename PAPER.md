# Performance Rating im Tischtennis
## Schätzung der tatsächlichen Spielstärke aus Satzergebnissen via Maximum-Likelihood

*Torsten Langer · BallSatzSieg · 2026*

---

## 1. Motivation

Das offizielle TTR-Rating (Tischtennis-Rating) misst langfristige Spielstärke. Es verändert sich langsam und reagiert ausschließlich auf das binäre Ergebnis einer Partie – Sieg oder Niederlage. Die eigentliche Qualität des Spiels, die sich in den Satzergebnissen widerspiegelt, bleibt unberücksichtigt.

Diese Arbeit stellt ein Verfahren vor, das aus den Satzergebnissen einer Saison oder eines Turniers ein **Performance Rating** berechnet: eine Schätzung der tatsächlichen Spielstärke, die unabhängig von der Siegbilanz ist. Ein knappes 2:3 gegen einen deutlich stärkeren Gegner kann dabei ein höheres Performance Rating ergeben als ein klares 3:0 gegen einen schwachen Gegner.

Die zentrale Frage lautet: *Auf welchem TTR-Niveau habe ich in dieser Saison tatsächlich gespielt?*

---

## 2. Modellarchitektur

Das Modell behandelt Tischtennis als stochastisches Spiel mit einer einzigen latenten Variable: der **Rally-Wahrscheinlichkeit** *p*, also der Wahrscheinlichkeit, einen einzelnen Punkt zu gewinnen. Alle weiteren Größen – Satzgewinn, Matchgewinn, Rating – folgen deterministisch aus *p* durch eine dreistufige Kaskade:

```
p  →  p_Satz  →  p_Match  →  TTR*
```

Das Modell geht dabei von keinerlei tischtennis-spezifischer Expertise aus. Es wird ausschließlich die Zählweise des Sports verwendet: wie viele Punkte ein Satz dauert, wie viele Sätze ein Match dauert, und wie das TTR-System Siegwahrscheinlichkeiten aus Ratingunterschieden berechnet. Ob ein Punkt durch Aufschlag, Topspin oder Fehler des Gegners zustande kam, spielt keine Rolle. Ebenso wenig fließen Faktoren wie Spielstil, Nervenstärke oder Tagesform in das Modell ein – all das ist in den Punktzahlen bereits enthalten, ohne explizit modelliert zu werden.

---

## 3. Die Kaskade

### 3.1 Rally → Satz

Ein Satz wird bis 11 Punkte gespielt, mit Verlängerung ab 10:10 bis zum 2-Punkte-Vorsprung. Die analytische Gewinnwahrscheinlichkeit eines Satzes bei Rally-Wahrscheinlichkeit *p* ist:

$$p_{\text{Satz}}(p) = \sum_{k=0}^{9} \binom{10+k}{10} p^{11} (1-p)^k + \binom{20}{10} p^{10}(1-p)^{10} \cdot \frac{p^2}{1 - 2p(1-p)}$$

Der erste Term summiert alle normalen Ausgänge (11:0 bis 11:9), der zweite Term behandelt die geometrische Reihe der Verlängerung ab 10:10.

### 3.2 Satz → Match

In einem Best-of-*N* Match (wobei *m* = ⌊*N*/2⌋ + 1 Sätze zum Sieg benötigt werden) gilt:

$$p_{\text{Match}}(p_{\text{Satz}}) = \sum_{k=0}^{m-1} \binom{m-1+k}{k} \, p_{\text{Satz}}^m \, (1 - p_{\text{Satz}})^k$$

Dies ist die negative Binomialverteilung für den ersten Spieler der *m* Sätze gewinnt.

### 3.3 Match → TTR (Inversion)

Das TTR-System definiert die Siegwahrscheinlichkeit von Spieler A (Rating TTR_A) gegen Spieler B (Rating *T*) als Funktion der Ratingdifferenz Δ = TTR_A − *T*:

$$p_{\text{TTR}}(\Delta) = \frac{1}{1 + 10^{-\Delta/150}}$$

Diese Formel wird nach TTR_A aufgelöst. Setzt man die aus der Kaskade berechnete Match-Wahrscheinlichkeit *p*_Match für *p*_TTR(Δ) ein und löst nach TTR_A = TTR\* auf, ergibt sich:

$$\text{TTR}^* = T - 150 \cdot \log_{10}\left(\frac{1}{p_{\text{Match}}} - 1\right)$$

wobei *T* das bekannte TTR des Gegners ist. TTR\* ist damit das Rating, das die beobachtete Match-Wahrscheinlichkeit exakt erklärt.

---

## 4. Maximum-Likelihood-Schätzung von p

### 4.1 Likelihood eines Satzergebnisses

Für ein Satzergebnis *a*:*b* (ohne Verlängerung, *a* > *b*) ist die Wahrscheinlichkeit:

$$\mathcal{L}(a{:}b \mid p) = \binom{a+b-1}{b} \cdot p^a \cdot (1-p)^b$$

Mit Verlängerung (beide Spieler bei 10 Punkten):

$$\mathcal{L}(a{:}b \mid p) = \binom{20}{10} p^{10}(1-p)^{10} \cdot (p(1-p))^{a-11} \cdot p^2$$

### 4.2 MLE über alle Sätze

Der MLE-Schätzer maximiert die gemeinsame Log-Likelihood über alle *K* Sätze einer Partie:

$$\hat{p} = \arg\max_p \sum_{i=1}^{K} \log \mathcal{L}(a_i{:}b_i \mid p)$$

Dies wird numerisch via `scipy.optimize.minimize_scalar` auf dem Intervall (0, 1) gelöst.

**Äquivalenz zu Method of Moments:** Da alle Sätze unabhängig sind und jede Rally ein Bernoulli-Versuch mit Parameter *p* ist, ist der MLE-Schätzer analytisch äquivalent zur einfachen Punktquote:

$$\hat{p}_{\text{MLE}} = \hat{p}_{\text{MoM}} = \frac{\sum_i a_i}{\sum_i (a_i + b_i)}$$

Die MLE-Formulierung ermöglicht jedoch die direkte Berechnung des Likelihood-Ratio-Tests (siehe Abschnitt 6).

---

## 5. Performance Rating über mehrere Partien

### 5.1 Einzelpartie

Für eine einzelne Partie gegen einen Gegner mit TTR-Wert *T* ergibt sich das Performance Rating direkt aus der Kaskadeninversion (vgl. Abschnitt 3.3):

$$\text{PR} = T - 150 \cdot \log_{10}\left(\frac{1}{p_{\text{Match}}(\hat{p})} - 1\right)$$

### 5.2 Saison (mehrere Partien)

Für *n* Partien gegen Gegner mit Ratings *T*₁, …, *T*ₙ wird das Performance Rating TTR\* als Lösung der Gleichung gesucht:

$$\sum_{i=1}^{n} \frac{1}{1 + 10^{-(\text{TTR}^* - T_i)/150}} = \sum_{i=1}^{n} p_{\text{Match},i}$$

Die linke Seite summiert die *erwarteten* Siege bei Rating TTR\*, die rechte Seite die *beobachteten* Match-Wahrscheinlichkeiten aus der Kaskade. Gelöst wird numerisch via Bisektionsverfahren (Brent-Methode).

Dies ist die Method-of-Moments-Formulierung auf Match-Ebene, die analytisch äquivalent zum MLE auf der kombinierten Rally-Ebene ist: die Information aus allen Rallies aller Partien fließt durch die Kaskade in den gemeinsamen Schätzer ein.

---

## 6. Modelldiagnose: Likelihood-Ratio-Test

Der χ²-basierte Likelihood-Ratio-Test prüft die Nullhypothese, dass *p* innerhalb einer Partie konstant war (H₀) gegen die Alternative, dass jeder Satz ein eigenes *p*ᵢ hatte (H₁):

$$D = 2 \sum_{i=1}^{K} \left[\log \mathcal{L}(a_i{:}b_i \mid \hat{p}_i) - \log \mathcal{L}(a_i{:}b_i \mid \hat{p})\right]$$

Unter H₀ gilt asymptotisch D ~ χ²(*K* − 1). Ein kleiner p-Wert deutet auf inkonsistente Spielstärke innerhalb der Partie hin und erhöht die Unsicherheit des Performance Ratings.

| p-Wert | Interpretation |
|--------|----------------|
| ≥ 10 % | konsistent |
| 5–10 % | auffällig |
| 1–5 %  | inkonsistent |
| < 1 %  | stark inkonsistent |

---

## 7. Konfidenzintervall via parametrischem Bootstrap

### 7.1 Methodik

Die Unsicherheit im Performance Rating entsteht ausschließlich durch die endliche Stichprobengröße der beobachteten Rallies. Pro Bootstrap-Iteration wird für jede Partie *i* die Rally-Wahrscheinlichkeit neu gezogen:

$$p_i^* \sim \text{Beta}(pp_i,\, op_i)$$

wobei *pp*ᵢ die gewonnenen und *op*ᵢ die verlorenen Rallies in Partie *i* sind. Dann wird die vollständige Kaskade durchlaufen und ein neues Performance Rating berechnet. Aus *B* = 2000 Iterationen werden die Quantile bestimmt.

### 7.2 Konsistenz mit dem Modell

Diese Formulierung ist konsistent mit der Grundphilosophie: *p* ist die atomare Einheit, alle anderen Größen sind deterministische Funktionen davon. Das Bootstrap-KI quantifiziert daher genau die Unsicherheit, die aus der endlichen Rally-Stichprobe entsteht – nicht mehr und nicht weniger.

### 7.3 Asymmetrie bei extremen Datensituationen

Bei stark einseitigen Serien – etwa 16 Siegen aus 16 Partien gegen deutlich schwächere Gegner – kann das Performance Rating außerhalb des 1σ-Konfidenzintervalls liegen. Die Ursache ist die Nichtlinearität der Kaskade: kleine Änderungen in *p* führen bei sehr großen *p*_Match-Werten zu überproportional großen Änderungen im TTR. Das Bootstrap-KI ist korrekt – der Punktschätzer ist robust, aber nicht median-unverzerrt in diesem Bereich.

---

## 8. Saisonverlauf und Trendanalyse

Wenn mehrere Punktspiele vorliegen, wird für jedes Punktspiel separat ein Performance Rating mit 1σ-Konfidenzintervall berechnet. Die Visualisierung zeigt:

- Punktspiel-Ratings mit asymmetrischen Fehlerbalken (1σ)
- Performance Rating der Gesamtsaison als horizontale Referenzlinie
- ±1σ und ±2σ des Gesamt-Konfidenzintervalls als gestrichelte Linien
- Bootstrap-Dichte (KDE) der Gesamtsaison als Hintergrundgradient

Ab drei Punktspielen kann ein **gewichteter linearer Trend** eingeblendet werden. Die Gewichtung erfolgt nach 1/σ²ᵢ der jeweiligen Punktspiel-Ratings – präzisere Schätzungen (mehr Spiele, mehr Rallies) zählen stärker. Das Konfidenzband des Fits wird analytisch berechnet (t-Verteilung mit *n* − 2 Freiheitsgraden).

---

## 9. Zusammenfassung

| Schritt | Methode | Formel |
|---------|---------|--------|
| Rally-Wahrscheinlichkeit | MLE | p̂ = Σaᵢ / Σ(aᵢ+bᵢ) |
| Satzwahrscheinlichkeit | Analytisch (Negative Binomial + Verlängerung) | p_Satz(p̂) |
| Matchwahrscheinlichkeit | Analytisch (Negative Binomial) | p_Match(p_Satz) |
| Performance Rating (Einzelpartie) | TTR-Inversion | TTR* = T − 150·log₁₀(1/p_Match − 1) |
| Performance Rating (Saison) | MoM (= MLE) | Σ p_TTR(TTR*,Tᵢ) = Σ p_Match,ᵢ |
| Konfidenzintervall | Parametrischer Bootstrap | Beta(ppᵢ, opᵢ) → Kaskade → TTR* |
| Modelldiagnose | Likelihood-Ratio-Test | D ~ χ²(K−1) |

Das Modell verwendet ausschließlich die Spielregeln und die TTR-Formel als Grundlage. Die einzige inhaltliche Annahme ist, dass *p* – die Wahrscheinlichkeit, einen einzelnen Punkt zu gewinnen – innerhalb einer Partie gleichbleibend ist. Diese Annahme wird durch den Likelihood-Ratio-Test (Abschnitt 6) explizit überprüft.

---

## Implementierung

Die App ist in Python/Streamlit implementiert und unter folgendem Link erreichbar:  
**https://tt-performance-rating.streamlit.app**

Quellcode: **https://github.com/torstenlanger/tt-performance-rating**

Abhängigkeiten: `streamlit`, `scipy`, `numpy`, `pandas`, `requests`, `beautifulsoup4`, `plotly`

---

## 10. Gleitender Verlauf

Neben dem einzelnen Punktspiel-Rating und dem linearen Trend bietet die App einen **gleitenden Verlauf**: für jedes Punktspiel *i* wird das Performance Rating aus einem rollierenden Fenster der letzten *w* ≤ 5 Punktspiele berechnet:

$$\text{PR}_i^{\text{roll}} = f\left(\bigcup_{j=\max(1,\,i-4)}^{i} \text{Spiele}_j\right)$$

Das Fenster wächst zu Saisonbeginn schrittweise an (1, 2, …, 5 Punktspiele) und bleibt ab dem fünften Punktspiel konstant bei maximal 5. Das Konfidenzintervall wird analog zum Gesamtrating per Bootstrap (500 Samples pro Fensterpunkt) berechnet und als 1σ- und 2σ-Band eingezeichnet.

Der gleitende Verlauf reagiert weniger empfindlich auf einzelne Ausreißer als die Einzelwerte und zeigt so den mittelfristigen Trend der Spielstärke.

---

## 11. LivePZ als Referenz

Zusätzlich zum Performance Rating kann der offizielle **LivePZ**-Verlauf als Referenzlinie eingeblendet werden. Der LivePZ nach einem Punktspiel ergibt sich aus:

$$\text{LivePZ}_{\text{nach}} = \text{TTR}_{\text{vor}} + \sum_{i=1}^{n} \Delta_i$$

wobei TTR_vor der Eingangs-TTR zu Beginn des Punktspiels ist (für alle Einzelspiele des Tages gleich) und Δ*ᵢ* die offizielle TTR-Veränderung des *i*-ten Einzelspiels. Der Vergleich zwischen LivePZ und Performance Rating gibt Aufschluss darüber, ob das offizielle Rating die tatsächliche Spielstärke über- oder unterschätzt.

---

## 12. Subset-Analysen

Bei Web-Import werden automatisch Teilmengen der Saison analysiert:

| Subset | Kriterium |
|--------|-----------|
| Vorrunde | Spiele der Hinrunde |
| Rückrunde | Spiele der Rückrunde |
| Heimspiele | H/G-Kennzeichen = „H" |
| Auswärtsspiele | H/G-Kennzeichen = „G" |
| Letzte 5 Punktspiele | Ab ≥ 6 Punktspielen verfügbar |

Jedes Subset enthält die vollständige Analyse: Performance Rating, 1σ- und 2σ-Konfidenzintervall, Bootstrap-Histogramm. Dies ermöglicht den Vergleich der Spielstärke unter verschiedenen Bedingungen – etwa ob Heim- und Auswärtsleistung signifikant voneinander abweichen.