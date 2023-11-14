## Informații relevante despre temă ##

* Capturi Tensorboard: tensorboard1.png, tensorboard2.png, tensorboard3.png + directorul runs
* Link WandB configuratii optimizatori (cu scopul de a se vedea eficienta lor): https://api.wandb.ai/links/me_myself_and_i/pi7lpsfu
* Proof - minim 3 configuratii pentru fiecare optimizator: minimum_3.png
* Link WandB peste 65% acuratete: https://api.wandb.ai/links/me_myself_and_i/muun8612; modelul poate fi vizualizat in main.py
* 65% acuratete se regaseste in cadrul checkpoint_5: reluarea parcursului de obtinere a acuratetii ar fi durat prea mult timp de rulare pentru a realiza grafice aferente fiecarei etape; mai intai, a fost obtinuta 60% acuratete cu o configuratie 4000ReLU - 1000Lin - 784Lin - 4000ReLU, cu SkipConnection pe blocul de dimensiune 784, dropout de 0.1 pe fiecare layer si optimizatorul SAM; apoi, acuratetea a ajuns la 63% prin extinderea modelului antrenat anterior cu inca un strat de 10Softmax, la care optimizatorul aplicat a fost Adagrad; au fost ajustate succesiv ratele de invatare cu gamma=0.5. La final, s-a obtinut 65% acuratete prin cresterea repetata a batch_size-ului de antrenare si schimbari de augmentare (la RandAugment).

##### Punctaj estimat: 8-9 puncte in functie de interpretarea ~65.5% vs 70% acuratete pentru ultimul bonus. ######