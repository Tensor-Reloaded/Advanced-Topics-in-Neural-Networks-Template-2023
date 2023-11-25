## Informații relevante despre temă ##

* am rezolvat atât partea obligatorie, cât și toate bonusurile (codul se găsește în folderul Assignment6) --> punctaj estimat: 9 temă (mie îmi ies 9 adunate la temă, nu 10, pe fișierul Assignment de pe Git) + 7 puncte bonus
* am realizat implementarea de mână a pasului de forawrd în rețele convoluționale (plus am obținut eroare foarte mică la testarea codului propus): fișierul *Handmade_Conv2d.py* (plus testarea în *main.py*)
* Runtime performance base model: 50 epoci în ~563 secunde
* am realizat partea de tracing, scripting și compiling pentru modelul propus de mine în *own_cnn.py*; comenzile pentru scripting și tracing se regăsesc în cadrul aceluiași fișier *own_cnn.py* în care este declarat modelul, înlocuindu-l pe acesta pentru etapa de forward a antrenării (vezi parametru suplimentar *model_forward* dat apoi la apelul *runner-ului* în cadrul aceluiași fișier); pentru partea de compiling am utilizat Google Colab și notebook-ul se regăsește aici: https://colab.research.google.com/drive/1z22tLsxueB3nTKrQoFnALBFZUbSWqHrh?usp=sharing
   * în ceea ce privește timpii de execuție, pentru tracing și scripting aceștia au scăzut cu aprox. 250 milisecunde pe epocă în medie (ținând cont și că nu am rulat foarte multe epoci pentru aceste experiment (50)); pentru compiling diferența a fost de 1146 secunde versus 1005 secunde (50 epoci) în favoarea modelului de bază, de unde pot concluziona că numărul de epoci ar mai trebui crescut pentru a se observa diferențe în favoarea modelului compilat
* am obținut peste 97% acuratețe la antrenare pe CIFAR10 atât cu modelul din *own_cnn.py*, cât și cu un model bazat pe PyramidNet (și augmentare cu CutMix) declarat în *main.py*; am utilizat modelul preantrenat ResNet50 din biblioteca pytorch pentru a obține peste 97% acuratețe (ceea ce s-a petrecut mai rapid decât la varianta fără utilizarea unui model preantrenat) prin finetuning cu:
   * arhitectura din *resnet_custom.py* (cu loss de tip CrossEntropy)
   * optimizatorul SGD cu *lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4*
   * sub 10 epoci de antrenare (2 pentru a depăși 97% acuratețe)
   * batch_size de 64
* aceleași modele (cel cu PyramidNet și cel preantrenat cu ResNet), adaptate doar ca set de date, au fost utilizate și pentru CIFAR100, pentru a depăși 75% acuratețe; pentru a dovedi rularea modelului pre-antrenat și pe setul de date CIFAR100 și a-l compara cu celălalt model, menționez aici câteva idei pornind de la primele 2 valori ale acurateții la validare obținute: 20.31% (mai puțin cu ~5% decât prima valoare pentru modelul cu PyramidNet) și 25.78% (se observă deja un pas mai mare de creștere, pasul de creștere a fost de aprox. 1% de-a lungul primelor epoci pentru modelul cu PyramidNet)
* *Observație CIFAR100: acuratețea se îmbunătățește mai rapid în cazul modelului preantrenat decât pentru modelul PyramidNet*

#### Fișiere-dovadă a rulărilor ####
* link wandb CIFAR10 model own_cnn: https://api.wandb.ai/links/me_myself_and_i/z2o7xbt5
* link wandb CIFAR10 model PyramidNet: https://api.wandb.ai/links/me_myself_and_i/p59t1vk9
* link wandb CIFAR10(/CIFAR100) model ResNet preantrenat: https://api.wandb.ai/links/me_myself_and_i/grxosc1p
* link wandb CIFAR100 model PyramidNet: https://api.wandb.ai/links/me_myself_and_i/qbjfrr7i
* document LATEX în Explanation.pdf
* plot acuratețe CIFAR10: plot_CIFAR10.png
* plot acuratețe CIFAR100: nu era specificat în cerințe explicit și plot pentru CIFAR100, dar este în curs de rulare (am tot așteptat să se termine, dar durează peste 10h modelul pre-antrenat pentru 75% acuratețe -- dar am realizat mai sus o scurtă comparație pentru acuratețile atinse între timp, dat fiind că în assignment nu se specifica un prag și pentru CIFAR100 la modelul preantrenat -- și nu am vrut să mai întârzii pull-request-ul)
* checkpoint inferență CIFAR10 model own_cnn: checkpoint_simpler (~40-45 epoci pentru pragul de 97%)
* checkpoint inferență CIFAR10 model PyramidNet: checkpoint_10_5 (ultimul număr reprezintă numărul necesar de epoci în care s-a atins pragul de acuratețe)
* checkpoint inferență CIFAR10 model ResNet preantrenat: checkpoint_10_2_p (>100MB; pot trimite ulterior prin link WeTransfer dacă îmi este solicitat)
* checkpoint inferență CIFAR100 model PyramidNet: checkpoint_100_78
* checkpoint inferență CIFAR100 model ResNet preantrenat: (>100MB; pot trimite ulterior prin link WeTransfer dacă îmi este solicitat)

*Notă: pentru rularea inferenței trebuie ca checkpoint-ul să fie plasat în directorul Assignment6 (ar trebui să folosesc WeTransfer pentru checkpoint-urile CIFAR100, deci ele nu ar fi deja acolo)*

