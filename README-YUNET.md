# eViacam — essai YuNet pour Windows

Cette branche ajoute un détecteur de visage neuronal à eViacam, avec une
comparaison reproductible hors caméra. Elle part du dépôt officiel
<https://github.com/cmauri/eviacam>, commit
`a4032ed9c59def5399a93e74f5ea84513d2f42b1`.

## Fonctionnement

1. `creavision` capture les images. Le port Windows CMake utilise DirectShow via
   OpenCV. L'énumération des caméras lit leurs noms sans les ouvrir.
2. `CVisionPipeline` confie une copie couleur à un thread de détection. Le réglage
   de consommation CPU conserve sa cadence habituelle (400 ms par défaut).
3. `FaceDetector` charge YuNet (modèle ONNX de mars 2023, OpenCV Zoo, licence MIT).
   Le traitement reste local, sans API ni envoi d'images. L'image est réduite
   seulement si son plus grand côté dépasse 640 pixels. Confiance minimale :
   0,85 ; suppression des doublons : 0,3 ; taille minimale de visage : 20 pixels.
4. Si l'image sombre ne donne aucun visage, une seconde passe applique CLAHE à
   sa luminance (Lab, limite 2, grille 8×8). Il ne s'agit pas de reconstruire des
   détails absents d'une image noire.
5. Le visage qui recoupe le précédent est privilégié ; sinon le plus grand
   visage est choisi. Ce mécanisme ne reconnaît pas l'identité d'une personne.
6. Le suivi Lucas–Kanade et les réglages de déplacement/clic d'eViacam restent
   utilisés. Une détection plus robuste n'implique pas automatiquement un
   pointeur plus précis si l'image est trop floue ou trop pauvre en détails.

La cascade Haar reste disponible avec `EVIACAM_FACE_DETECTOR=haar`. Ses paramètres
de référence sont ceux du dépôt : facteur 1,5, deux voisins et minimum 65×65.
Elle sert également de secours si YuNet ne peut pas être chargé/exécuté.
Le titre de la fenêtre indique le moteur réellement actif.

## Autres corrections nécessaires

- Synchronisation de la copie d'image, de la publication du rectangle et du réveil
  du thread ; arrêt avec attente du thread effectivement démarré.
- Résultats vieux de plus de 500 ms écartés ; rectangles bornés à l'image.
- Petites régions compatibles avec les visages éloignés ; initialisation lors
  d'un changement de résolution ; sortie sûre sur une image invalide.
- Correction du réglage CPU « faible », auparavant ramené à « minimum ».
- Chargement des préférences avant le démarrage de la capture.
- Configuration portable séparée du registre de la version installée.
- Port Windows CMake/GCC 64 bits, sans l'ancien SDK DirectShow de Visual Studio 2013.

## Compilation reproductible

Versions utilisées : MSYS2 UCRT64/GCC 16.2, wxWidgets 3.2.11 et OpenCV 4.10.0
compilé depuis ses sources. Le paquet OpenCV 5 actuel de MSYS2 n'est pas utilisé
pour l'application : l'ancien code d'eViacam emploie encore des API C supprimées.

Utiliser des chemins de compilation sans accent pour GCC/ld. Des jonctions
`Documents/eviacam-source` et `Documents/eviacam-build-tools` peuvent pointer
vers le dépôt et la chaîne de compilation. Le dossier physique de la chaîne
doit lui aussi avoir un nom sans accent.

Dans MSYS2, installer `mingw-w64-ucrt-x86_64-gcc`,
`mingw-w64-ucrt-x86_64-cmake`, `mingw-w64-ucrt-x86_64-ninja` et
`mingw-w64-ucrt-x86_64-wxwidgets3.2-msw`.

Extraire <https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz>
dans `.deps/opencv-4.10.0`, puis lancer depuis PowerShell :

```powershell
./tools/build-opencv.ps1
# Python 3.10, idéalement dans un environnement dédié
python -m pip install -r tools/requirements-benchmark.txt
python tools/prepare_benchmark.py
./tools/build-windows.ps1
./build/face_benchmark.exe . tests/generated/manifest.tsv tests/generated/results.csv
python tools/summarize_benchmark.py
./tools/package-portable.ps1
```

Les scripts acceptent `-Toolchain` pour changer de chaîne de compilation. Les DLL
d'OpenCV et du compilateur doivent être dans `PATH` lors des tests depuis le dépôt.
Le paquet portable copie récursivement les DLL importées et leurs licences.

Modèle :
<https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>

SHA-256 du fichier `face_detection_yunet_2023mar.onnx` :
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.

## Validation sans caméra

`detector_tests` vérifie le chargement neuronal, le mode Haar, le secours en cas
de modèle manquant, les images vides/noires/trop petites, les limites des
rectangles, une détection réelle et la continuité de sélection.

`pipeline_tests` exécute le vrai pipeline asynchrone et son suivi : acquisition
d'un petit visage sombre, disparition et arrêt du mouvement autorisé après le
délai configuré, changements de résolution, image vide, réglage CPU et arrêt.
Il n'instancie pas de caméra ni de contrôleur de souris.

`face_benchmark` appelle exactement le détecteur lié à l'application. Les deux
moteurs utilisent la même compilation d'OpenCV, deux threads CPU et les mêmes
images de 320×240 pixels. Les temps incluent le prétraitement mais excluent le
chargement du modèle, la lecture du fichier et l'échauffement initial.

Les 126 cas comprennent 96 images avec un visage et 30 images sans visage.
Les positifs proviennent de **trois portraits seulement**, avec des boîtes
annotées manuellement, des largeurs de 24 à 120 pixels, quatre niveaux de lumière,
du bruit, du flou, un gamma modifié et un éclairage latéral simulé. Une détection
est correcte si son intersection sur union avec l'annotation atteint 0,3.
Les transformations sont corrélées : ce jeu est un test de résistance ciblé,
pas une mesure représentative de toutes les personnes et webcams.

Les critères sont définis dans `tools/summarize_benchmark.py` : gain global d'au
moins 20 points, aucune baisse sur les cas de référence, gain en lumière et en
distance, pas d'augmentation des faux positifs sur les négatifs et 95e percentile
inférieur à la période de détection par défaut de 400 ms.

Les résultats ne comparent pas directement l'exécutable installé de 2016 avec
OpenCV 2.4. Ils comparent son approche Haar, aux paramètres du dépôt actuel,
à YuNet dans le même environnement. Le confort réel, la webcam, l'exposition,
les mouvements rapides, les occlusions et la fiabilité du retour vers l'ancienne
application doivent encore être vérifiés lors d'un essai utilisateur.

## Essai utilisateur

Les lanceurs de contrôle réel demandent maintenant une confirmation UAC, comme
l'exécutable eViacam installé (`requireAdministrator`). Le premier paquet
démarrait YuNet avec les droits ordinaires : le pointeur pouvait se bloquer sur
certaines interfaces Windows, puis repartir dans une fenêtre normale. Cette
différence de droits est corrigée dans les lanceurs. Le 31 octobre 2026,
l'utilisateur a confirmé que la version corrigée fonctionne lors de son essai,
y compris après le problème de barre des tâches signalé. Ce retour reste une
validation sur un poste, pas une garantie sur toutes les configurations Windows.
La démonstration reste sans élévation. Aucune politique de sécurité n'est modifiée.
`4 - Lancer IA barre des taches.vbs` ouvre une session sans limite de temps et
relance l'ancien logiciel quand YuNet se ferme. Les essais 2 et 3 conservent leur
limite de deux minutes. Lancer directement `eviacam.exe` n'applique pas cette
correction des droits.

Le dossier portable contient une démonstration et deux essais limités à deux
minutes (YuNet et Haar). Le lancement d'un essai attend que l'utilisateur ferme
sa version habituelle, puis ouvre la nouvelle avec une copie de ses réglages.
Il relance l'ancienne après deux minutes ou à la fermeture de l'essai.
La version installée et ses préférences ne sont pas remplacées.

`EVIACAM_REPLAY=chemin.avi` sélectionne une vidéo au lieu de la caméra et bloque
les commandes de souris au point d'envoi. `EVIACAM_TIMED_TRIAL=1` permet au
superviseur de fermer uniquement son processus d'essai sans dialogue de sortie.
Une démonstration n'est donc jamais un essai de contrôle réel du pointeur.

Le lanceur dépend de Windows PowerShell et de Windows Script Host. Le retour
automatique est une précaution pratique, pas une garantie contre une panne de
Windows, un blocage de sécurité ou l'arrêt du superviseur par l'utilisateur.
