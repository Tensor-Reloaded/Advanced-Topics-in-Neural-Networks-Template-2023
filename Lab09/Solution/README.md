## Informații relevante despre temă -- Rezumat document LaTeX ##

* Punctaj estimat: 10 puncte
* Modele folosite: un model cu 2 straturi fully-connected de forma 1024 * 3 - 128 - 784 (prima încercare de rezolvare, inspirată din ce am folosit pentru tema 4 cu imaginile din satelit) --- double_linear.py --- și un model cu 4 straturi convoluționale (inspirat parțial din lucrarea https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8456517) pentru care am introdus perceptual loss (pe baza VGG16) și am observat și notat în documentul LaTeX îmbunătățiri semnificative față de utilizarea MSELoss pe același model --- cnn_own.py. Adițional, am experimentat un model cu un singur strat liniar în urma discuției de la seminar (main.py).
* Performanța îmbunătățită față de transformările efectuate secvențial pe CPU are loc doar în cazul primului model, dar cel de-al doilea rămâne de interes pentru deprinderea cu un nou tip de loss și pentru performanța transformărilor utilizând loss-ul perceptual (deși aceeași performanța este atinsă de primul model într-un timp cu mult redus).
* Loss folosit în cadrul primului model: MSELoss.
* Criteriu de oprire: loss-ul nu a scăzut cu mai mult de 0.000001 în ultimele 10 epoci (prag ales pe bază de observații experimentale) pe setul de validare, efectuând verificarea la fiecare epocă față de epoca curentă - 10 epoci.

#### Notă: am continuat să folosesc framework-ul implementat la laboratoarele precedente și am încercat să păstrez echivalența cu ce se afla în fișierul main.py de pe Git. ####
