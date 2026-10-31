# eViacam avec détection de visage YuNet

La branche `master` contient la version Windows compilée et essayée le 31 octobre
2026, ainsi qu'un premier port natif macOS pour Apple Silicon et Intel (macOS 15
minimum). Les deux utilisent la détection neuronale locale YuNet et le suivi
eViacam, avec des adaptations pour la caméra et le pointeur de chaque système.

- [Fonctionnement, compilation et utilisation](README-YUNET.md)
- [Version macOS : installation, autorisations et compilation](README-MACOS.md)
- [Résultats, protocole et limites des tests](RESULTATS-YUNET.md)
- [Sources du détecteur](src/facedetector.cpp)
- [Tests sans caméra](tests/)
- [Scripts de compilation et de distribution](tools/)

La comparaison hors caméra obtient 82 détections correctes sur 96 cas avec
YuNet, contre 36 avec Haar. Il s'agit de variantes synthétiques de trois portraits,
pas d'une évaluation représentative de toutes les webcams ou personnes.
L'utilisateur a confirmé le fonctionnement après correction du lancement pour
la barre des tâches Windows.

Sur Windows, pour le contrôle réel, utiliser les lanceurs VBS générés par
`tools/package-portable.ps1`, puis accepter la confirmation Windows.
Le lanceur 4 ouvre une session sans limite de temps. La démonstration n'utilise
pas la webcam et ne commande pas le pointeur.

Cette variante part du [projet original de César Mauri](https://github.com/cmauri/eviacam)
au commit `a4032ed9c59def5399a93e74f5ea84513d2f42b1`. Elle remplace la variante
MediaPipe précédente dans `master`. L'historique de cette variante et du projet
original est conservé, ainsi que la page de téléchargement dans `docs/`.
Les anciennes releases restent des versions antérieures. Les builds YuNet sont
disponibles comme artefacts des workflows Windows Build et macOS Build ; les
versions macOS à télécharger sont publiées en préversion dans
[Releases](https://github.com/Mohamed-Ali-SOBHI/eviacam/releases).

eViacam est distribué sous [GPL version 3 ou ultérieure](COPYING).
Le modèle YuNet est fourni avec sa [licence MIT](models/LICENSE-YuNet).
Les paramètres personnels, captures de webcam, dépendances téléchargées et
fichiers de compilation locaux ne font pas partie de cette branche.
