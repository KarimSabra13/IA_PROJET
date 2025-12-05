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
