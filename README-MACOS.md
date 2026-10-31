# eViacam YuNet pour macOS

Port natif Cocoa / AVFoundation / Quartz, avec détection YuNet locale. Deux
archives distinctes : Apple Silicon (`arm64`) et Intel (`x86_64`), macOS 15 ou
plus récent. La compilation et les tests sont exécutés sur les deux architectures
dans [GitHub Actions](https://github.com/Mohamed-Ali-SOBHI/eviacam/actions/workflows/macos-build.yml).

## Installation et autorisations

Extraire l'archive correspondant au processeur et déplacer `eViacam-YuNet.app`
dans Applications avant le premier lancement. Aucune installation de Python,
OpenCV, wxWidgets ou Homebrew n'est nécessaire pour utiliser l'application.

Cette version de test possède une signature locale ad hoc, sans certificat
Apple Developer ID ni notarisation. Si macOS bloque le premier lancement,
utiliser « Ouvrir quand même » dans Réglages Système → Confidentialité et sécurité,
après avoir vérifié la provenance du téléchargement. Ne pas désactiver Gatekeeper.

Autoriser la **caméra** lors du premier lancement. Cliquer sur Activer pour
commander le pointeur : macOS demande l'autorisation **Accessibilité** pour
eViacam YuNet. L'accorder dans Réglages Système → Confidentialité et sécurité →
Accessibilité, puis cliquer à nouveau sur Activer. Les raccourcis globaux sont
facultatifs, désactivés par défaut, et nécessitent l'autorisation Surveillance
de l'entrée si on les active dans les options.

Les réglages sont enregistrés dans le dossier utilisateur de l'application,
sous `~/Library/Application Support/`, jamais à l'intérieur du `.app`.
La version macOS ne demande pas les droits administrateur.

## Portée des tests

Le workflow compile l'application, exécute les tests du détecteur, du pipeline
et des événements Quartz (déplacement, glisser, relâchement, double clic),
puis lance le véritable `.app` Cocoa pendant 90 images d'une vidéo de test.
Ce test vérifie le chargement des ressources, YuNet, l'interface et la fermeture,
sans ouvrir de caméra ni envoyer de mouvements ou de clics. Il vérifie aussi que
l'exécutable ne dépend que des bibliothèques système macOS.

L'acceptation des autorisations, la webcam réelle, le déplacement/clic/glisser et
le confort doivent encore être vérifiés sur un Mac réel. La zone de travail est
limitée à l'écran principal. La barre de clics est un panneau flottant ; elle ne
réserve pas de bande sur le bureau et son masquage automatique n'est pas implémenté.
Le démarrage automatique et le lancement d'un clavier visuel ne sont pas fournis
par ce premier port macOS.

## Premier essai sur un Mac

1. Dans « À propos de ce Mac », vérifier le processeur et la version de macOS,
   puis choisir l'archive correspondante (Apple Silicon ou Intel, macOS 15 minimum).
2. Après les autorisations et l'assistant de réglage, vérifier que la vidéo
   apparaît et que le cadre suit le visage à distance normale, puis plus loin
   et avec une lumière plus faible.
3. Activer le contrôle et essayer le déplacement, le clic, le double clic et
   le glisser-déposer, notamment dans le Finder et le Dock.
4. Sortir le visage du champ, vérifier que le pointeur s'arrête, puis revenir
   devant la caméra et vérifier la reprise.
5. Fermer et relancer l'application pour vérifier la conservation des réglages.

En cas de problème, relever le modèle du Mac, la version de macOS, le type de
webcam et l'étape concernée. Ces essais matériels complètent les tests sans caméra.

## Compilation

Avec Xcode Command Line Tools, CMake, Ninja et Python 3.10 :

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r tools/requirements-benchmark.txt
python tools/prepare_benchmark.py
bash tools/build-macos-deps.sh
bash tools/build-macos.sh
bash tools/package-macos.sh
```

Les scripts construisent wxWidgets 3.2.11 et OpenCV 4.10.0 localement et les lient
statiquement. L'archive produite dans `dist/` contient l'application et les licences.
Le modèle YuNet et les limites du protocole comparatif sont documentés dans
[README-YUNET.md](README-YUNET.md) et [RESULTATS-YUNET.md](RESULTATS-YUNET.md).
