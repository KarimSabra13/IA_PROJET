Question 1 – Filtre RC du premier ordre

Objectif
Décrire un filtre RC du premier ordre en SPICE, avec des paramètres pour R et C, puis calculer la fréquence de coupure théorique.
Le netlist complet du filtre est dans le fichier rc_filter.cir.
Description du circuit
On modélise un filtre RC passe-bas:
- Source Vin appliquée sur une résistance R.
- Un condensateur C entre la sortie et la masse.
- La sortie se trouve entre R et C.

Le netlist SPICE utilise des paramètres pour R et C:

Filtre RC (ngspice)

Rval = valeur de la résistance
Cval = valeur de la capacité

Netlist:

* Filtre RC passe-bas du premier ordre

.param Rval = 1k
.param Cval = 100n

Vin in 0 AC 1
R1 in out {Rval}
C1 out 0 {Cval}

.ac dec 100 10 1Meg
.end

Fréquence de coupure théorique

Formule:
fc = 1 / (2 * pi * R * C)

Avec:
R = 1 kΩ = 1e3 ohms
C = 100 nF = 100e-9 farads

Produit RC:
RC = 1e3 * 100e-9 = 1e-4 secondes

Donc:
fc = 1 / (2 * pi * 1e-4)
fc ≈ 1 / (6.283e-4)
fc ≈ 1.59e3 Hz

Fréquence de coupure théorique du filtre:
fc ≈ 1.6 kHz.





## Question 2 – Source de tension et simulation AC

Objectif  
Ajouter une source de tension sur l’entrée du filtre et configurer une simulation AC pour pouvoir extraire la fréquence de coupure.

Choix de la source  
On utilise une source de tension linéaire Vin entre le nœud `in` et la masse, avec une amplitude AC de 1 V.  
En SPICE, le mot-clé `AC` définit l’amplitude pour l’analyse fréquentielle petit signal.

Ligne correspondante dans le netlist:

```spice
Vin in 0 AC 1



Question 3 – Mesure de la fréquence de coupure avec .meas

Objectif
Automatiser le calcul de la fréquence de coupure directement dans ngspice.

Principe
Pour une source AC de 1 V, le gain en décibels vaut:

gain_dB = vdb(out)

La fréquence de coupure correspond au point où le gain vaut −3 dB.
La directive .meas suivante retourne la fréquence recherchée:

.meas ac f_cutoff WHEN vdb(out) = -3

Explication des mots clés

.meas ac : demande une mesure sur l’analyse AC.

f_cutoff : nom du résultat qui apparaîtra dans la console ngspice.

WHEN vdb(out) = -3 : ngspice recherche la fréquence pour laquelle vdb(out) atteint −3 dB en interpolant entre les points du sweep AC.

Utilisation

Lancer ngspice sur le fichier:

ngspice rc_filter.cir

À la fin de l’analyse, ngspice affiche une ligne du type:

f_cutoff = 1.59e+03

Cette fréquence se compare ensuite à la valeur théorique 1,6 kHz obtenue à la question 1.
