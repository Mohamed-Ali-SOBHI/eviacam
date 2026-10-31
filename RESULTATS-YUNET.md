# Validation du 31 octobre 2026

Compilation Windows 64 bits réussie : eViacam basé sur le commit officiel
`a4032ed9c59def5399a93e74f5ea84513d2f42b1`, wxWidgets 3.2.11, OpenCV 4.10.0
recompilé localement et YuNet de mars 2023. Machine : Intel Core i7-12700H.

## Comparaison du code C++ utilisé par l'application

| Scénario | Haar du dépôt | YuNet ajouté |
|---|---:|---:|
| Ensemble des images avec visage | 36/96 (37,5 %) | 82/96 (85,4 %) |
| Éclairage réduit, visage de 96 ou 120 pixels | 12/18 | 18/18 |
| Distance seule, visage de 24 à 64 pixels | 3/12 | 9/12 |
| Distance et éclairage réduit combinés | 6/36 | 28/36 |
| Cas de référence | 6/6 | 6/6 |
| Gamma modifié | 2/6 | 6/6 |
| Éclairage latéral simulé | 3/6 | 6/6 |
| Bruit ajouté | 2/6 | 4/6 |
| Flou ajouté | 2/6 | 5/6 |
| Faux positifs sur 30 images sans visage | 0 | 0 |
| Détections supplémentaires ou mal localisées, sur tout le jeu | 10 | 0 |

YuNet gagne **47,9 points** de détection sur ce jeu. Il rate encore 14 cas ;
ce résultat ne signifie donc pas que la détection est parfaite.

| Temps d'une analyse, prétraitement inclus | Haar | YuNet |
|---|---:|---:|
| Moyenne | 1,42 ms | 12,87 ms |
| Médiane | 1,51 ms | 8,51 ms |
| 95e percentile | 2,90 ms | 35,96 ms |

Le gain de détection coûte plus de calcul. La détection s'exécute sur son thread
de travail, à la cadence configurée (400 ms par défaut), tandis que le suivi
de points continue sur les images de la caméra. Ces durées ne sont pas une
mesure de latence totale du pointeur ni une garantie de fréquence webcam.

Les six critères d'acceptation du script de synthèse sont satisfaits.
Les deux exécutables de test passent : chargement/fallback/sélection et pipeline
asynchrone avec acquisition, translation connue, disparition, délai d'arrêt,
changements de résolution et fermeture du thread. La distribution a également
chargé et testé le détecteur avec un PATH limité aux dossiers système Windows.
La démonstration a été ouverte et vérifiée visuellement à 30 images/s, avec
le suivi actif sous éclairage simulé réduit et sans dialogue d'erreur. Le
lanceur VBS fourni a démarré la démonstration ; sa fermeture a aussi terminé
son superviseur PowerShell. Le processus eViacam habituel est resté ouvert.

SHA-256 de l'exécutable livré :
`aa2ab8fcc0e2c003351e7b5d1f3fd9eb5b708f65115f5559e0bc032bfd1ab384`.

## Portée et limites

Retour utilisateur du 31 octobre 2026 : premier essai réel positif, puis
signalement d'un blocage du pointeur sur la barre des tâches. Les lanceurs de
contrôle réel ont été corrigés pour demander les mêmes droits administrateur
que l'eViacam original. L'utilisateur a ensuite confirmé le bon fonctionnement.
Cette validation manuelle sur un poste complète les tests hors caméra ci-dessus.

- 126 images de 320×240, dont 96 variantes corrélées de trois portraits publics.
  Boîtes de référence annotées manuellement ; détection validée à IoU ≥ 0,3.
- Même compilation d'OpenCV et deux threads CPU pour les deux détecteurs ;
  échauffement avant les mesures. Ordinateur utilisé normalement pendant le test.
- Le mode Haar reprend exactement les paramètres du dépôt actuel. Il ne s'agit
  pas d'une comparaison directe avec le binaire installé de 2016/OpenCV 2.4.
- Aucun réglage de seuil n'a été ajusté après lecture du résultat comparatif.
- Les tests automatisés n'ont pas ouvert la webcam occupée. Le retour utilisateur
  ci-dessus ne constitue pas une mesure systématique du confort, des mouvements
  de tête, de l'exposition matérielle, des occlusions ou du retour automatique
  entre les deux applications.
- Le contrôle visuel utilise une vidéo synthétique animée à partir d'un portrait,
  pas une vidéo réelle de mouvements de tête. Le mode vidéo bloque l'envoi de clics
  et de mouvements à Windows.

Les fichiers bruts sont dans `tests/generated/results.csv`, `summary.json`,
`manifest.tsv` et `provenance.json`. La génération et le protocole sont décrits
dans `README-YUNET.md`.
