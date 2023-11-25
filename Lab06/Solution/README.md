## Informații relevante despre temă ##

* am rezolvat atât partea obligatorie, cât și toate bonusurile (codul se găsește în folderul Assignment6) --> punctaj estimat: 9 + 7 puncte
* am realizat implementarea de mână a pasului de forawrd în rețele convoluționale (plus am obținut eroare foarte mică la testarea codului propus): fișierul *Handmade_Conv2d.py* (plus testarea în *main.py*)
* Runtime performance base model: 50 epoci în ~  secunde
* am realizat partea de tracing, scripting și compiling pentru modelul propus de mine în *own_cnn.py*; comenzile pentru scripting și tracing se regăsesc în cadrul aceluiași fișier *own_cnn.py* în care este declarat modelul, înlocuindu-l pe acesta pentru etapa de forward a antrenării (vezi parametru suplimentar *model_forward* dat apoi la apelul *runner-ului* în cadrul aceluiași fișier); pentru partea de compiling am utilizat Google Colab și notebook-ul se regăsește aici: https://colab.research.google.com/drive/1z22tLsxueB3nTKrQoFnALBFZUbSWqHrh?usp=sharing
   * în ceea ce privește timpii de execuție, pentru tracing și scripting aceștia au scăzut în general cu aprox. 1 secundă pe epocă (ținând cont și că nu am rulat foarte multe epoci pentru aceste experiment); pentru compiling a trebuit să cresc destul de mult numărul de epoci pentru a se vedea o diferență
* am obținut peste 97% acuratețe la antrenare pe CIFAR10 atât cu modelul din *own_cnn.py*, cât și cu un model bazat pe PyramidNet (și augmentare cu CutMix) declarat în *main.py*; am utilizat modelul preantrenat ResNet50 din biblioteca pytorch pentru a obține peste 97% acuratețe (ceea ce s-a petrecut mai rapid decât la varianta fără utilizarea unui model preantrenat) prin finetuning cu:
   * arhitectura din *resnet_custom.py* (+ loss de tip BCE)
   * optimizatorul SGD cu *lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4*
   * sub 10 epoci de antrenare (2 pentru a depăși 97% acuratețe)
* aceleași modele (cel cu PyramidNet și cel preantrenat cu ResNet) au fost utilizate și pentru CIFAR100, pentru a depăși 75% acuratețe

#### Fișiere-dovadă a rulărilor ####
* link wandb CIFAR10 model own_cnn:
* link wandb CIFAR10 model PyramidNet:
* link wandb CIFAR10 model ResNet preantrenat:
* link wandb CIFAR100 model PyramidNet:
* link wandb CIFAR100 model ResNet preantrenat: în curs de rulare
* document LATEX în Explanation.pdf
* plot acuratețe CIFAR10:
* plot acuratețe CIFAR100: în curs de rulare; *Observație: acuratețea se îmbunătățește mai rapid pentru modelul PyramidNet*
* checkpoint inferență CIFAR10 model own_cnn: checkpoint_simpler (~45-50 epoci pentru pragul de 97%)
* checkpoint inferență CIFAR10 model PyramidNet: checkpoint_10_31 (ultimul număr reprezintă numărul necesar de epoci în care s-a atins pragul de acuratețe)
* checkpoint inferență CIFAR10 model ResNet preantrenat: checkpoint_10_2_p
* checkpoint inferență CIFAR100 model PyramidNet: checkpoint_100_78
* checkpoint inferență CIFAR100 model ResNet preantrenat: în curs de rulare

