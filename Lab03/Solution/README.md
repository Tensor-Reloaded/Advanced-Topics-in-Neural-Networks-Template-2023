## Informații relevante despre temă ##

* am rezolvat tema cu obținerea a peste 95% acuratețe -- în mai puțin de 20 de epoci (în mai puțin de 10 epoci dacă se folosește inclusiv implementarea optimizatorului Adam din cadrul bonusului). Pentru mai multe epoci, modelul creat se situează la peste 96% acuratețe.
* pentru temă am testat și, eventual, păstrat tehnicile de weight decay, learning rate decay, batch normalization, weight initialization (am încercat și dropout dar nu mi s-a părut că rezultatele sunt notabile) 
* pentru activarea stratului ascuns am folosit ReLU (partea în care am încercat cu softmax a rămas comentată ca încercare)
* pentru bonus am adăugat o implementare a optimizatorului Adam; am schimbat și arhitectura rețelei în 2 straturi ascunse a câte 512 neuroni.
