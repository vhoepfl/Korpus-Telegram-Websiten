# Embedding-basierte semantische Analyse von Telegram-Korpus

In diesem Repository gibt es zwei verschiedene Ansätze: 
### Embeddings vollständig auf Telegram-Korpus trainieren (siehe create_embeddings.ipynb)
Vorteil: kontextuelle Bedeutungen vollständig erfasst. <br>
Nachteil: Relativ viel noise bei kleinen Trainingskorpora. 
Noise zeigt sich beispielsweise darin, dass lexikalisch ähnliche Wörter nahe beieinander sind, da nicht genug Information vorhanden ist, um diese trotz eines großen Bedeutungsunterschieds zu trennen, solang es hohe Überschneidung in den subwords gibt. 

### Pretrained Embeddings mit Finetuning auf Telegram-Korpus
Das Grundprinzip hier ist differenzielle Analyse: Jene Begriffe, die sich durch Finetuning am stärksten im Embedding-space verschieben, sind potentiell relevant. <br>
Das bedeutet, dass die gefinetunten Embeddings in erster Linie das sehr generelle pretraining-Datenset repräsentieren, mit nur leichter Anpassung an das Telegram-Korpus.     Konkret wurden pretrained [fastText](https://fasttext.cc/docs/en/language-identification.html)-Embeddings für Französisch verwendet, die eine hohe Zahl an Wörtern in hoher Qualität repräsentieren. <br>
Finetuning eines vortrainierten fastText-Modells: `./fasttext skipgram -input training_corpus.txt -output cc.fr.finetuned.300.bin -epoch 1 -dim 300 -pretrainedVectors cc.fr.300.bin`<br>

*@Laura: Die Embeddings sind mind. 5GB groß, und da sowohl das pretrained als auch das gefinetunte Embedding geladen werden muss, wird das am Laptop vermutlich recht schwierig, daher sind die hier nicht mit hochgeladen.*

Es gibt die folgenden Möglichkeiten, relevante Wörter zu identifizieren: 
- **SHIFT:** Wörter mit größter Bedeutungsverschiebung
    In der Praxis vermutlich nahezu nutzlos, da Verschiebungen in hochdimensionalen Räumen sehr schnell sehr groß werden können, ohne dass dies auf signifikante semantische Verschiebungen hindeutet. 
    Könnte ausgeführt werden mit `python measure_shifts.py --mode=shift <pretrained_embeddings>.bin <finetuned_embeddings>.bin -k 200 -o test_shift.csv`<br

- **TOWARDS:** Wörter, die sich am stärksten an ein gegebenes Zielwort annähern. 
    - Zielwörter direkt übergegeben: 
        ` python measure_shift.py <pretrained_embeddings>.bin <finetuned_embeddings>.bin --mode toward --targets <Zielwort 1> <Zielwort 2> --min-count 20 5 -k 100 -o <Output-Dateiname>`
    - Zielwörter in Datei: 
        `python measure_shift.py <pretrained_embeddings>.bin <finetuned_embeddings>.bin --mode toward --targets-file <target_file> --min-count 20 5 -k 100 -o <Output-Dateiname>`
    Implementiert mittels Z-Score: Jede Bewegung eines Worts relativ zu einem Zielwort wird skaliert mittels der Standardabweichung. 
    Standardabweichung ist vorzeichenlos, misst also nur Stärke der Bewegung, Richtung ist egal. Da alle Wörter mit Bewegung in Richtung Zielwort ein positives Vorzeichen erhalten, besitzen Wörter, die sich überdurchschnittlich in Richtung Zielwort bewegen, die höchsten Scores. <br>
    Konkret wird die Richtung der Bewegung per Kosinusähnlichkeit gemessen, d.h. nicht mit einem echten Richtungsmaß wie euklidischer Distanz, da das in einem hochdimensionalen Raum wie 300-dimensionalem Embedding-space nicht mehr gut funktioniert. 
    **Globale Standardabweichung:** Bewegung aller Wörter relativ zu Zielwort t<br>
    Diese Metrik findet generell Wörter, die sich überdurchschnittlich an Zielwort t annähern. 
    **Lokale Standardabweichung:**  Für jedes Wort w: Bewegung der k (mit k=50) nächsten Wörter für w. <br>
    Damit findet diese Metrik Wörter w, die sich aus ihrer lokalen Umgebung entfernen, und eine neue, kontextspezifische Bedeutung näher an Zielwort t annehmen. <br><br>
    
    Mithilfe des Parameters `loc-glob-ratio`(funktioniert mit Werten von 0-1) kann das Verhältnis von globalem zu lokalem Kontext eingestellt werden. Für nur globalen Kontext auf 0 setzen, für nur lokalen Kontext auf 1. Standardeinstellung, falls dieser Parameter wegelassen, ist 0.7. Nur globalen Kontext zu verwenden, ist *massiv* schneller als eine Mischung oder nur lokale Information. <br>




