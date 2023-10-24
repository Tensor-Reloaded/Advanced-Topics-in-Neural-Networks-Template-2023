## Informații relevante despre temă ##

* am rezolvat tema cu obținerea a peste 95% acuratețe -- în mai puțin de 20 de epoci (în mai puțin de 10 epoci dacă se folosește inclusiv implementarea optimizatorului Adam din cadrul bonusului). Pentru mai multe epoci, modelul creat se situează la peste 96% acuratețe (parte din bonus).
* pentru temă am testat și, eventual, păstrat tehnicile de data augmentation, weight decay, learning rate decay, batch normalization, weight initialization (am încercat și dropout dar nu mi s-a părut că rezultatele sunt notabile) 
* pentru activarea stratului ascuns am folosit ReLU (partea în care am încercat cu softmax a rămas comentată ca încercare)
* cu îmbunătățirea procesului de data augmentation prin adăugarea mai multor tipuri de rotații și combinații de rotații + shiftări, se ajunge la 95% acuratețe la varianta de bonus în 4 epoci
* pentru bonus am adăugat o implementare a optimizatorului Adam; am mai schimbat și arhitectura rețelei în 2 straturi ascunse a câte 512 neuroni.
