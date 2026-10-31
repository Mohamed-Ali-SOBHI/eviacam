// Downloads and instructions remain usable without JavaScript.
const motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
const motionToggle = document.getElementById("motion-toggle");
let motionPaused = motionPreference.matches;
function updateMotion() {
  document.body.classList.toggle("motion-paused", motionPaused);
  motionToggle.textContent = motionPaused ? "Animer" : "Pause";
  motionToggle.setAttribute(
    "aria-label",
    motionPaused ? "Animer l’illustration" : "Mettre l’illustration en pause",
  );
  motionToggle.setAttribute("aria-pressed", String(motionPaused));
  motionToggle.hidden = motionPreference.matches;
}
motionToggle.addEventListener("click", () => {
  motionPaused = !motionPaused;
  updateMotion();
});
motionPreference.addEventListener("change", () => {
  motionPaused = motionPreference.matches;
  updateMotion();
});
updateMotion();
const toast = document.getElementById("copy-toast");
let toastTimer;
document.querySelectorAll("[data-copy-hash]").forEach((button) => {
  if (!navigator.clipboard?.writeText) return;
  button.hidden = false;
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(button.dataset.copyHash);
      toast.textContent = "Empreinte SHA-256 copiée.";
    } catch {
      toast.textContent =
        "Copie indisponible : sélectionnez l’empreinte affichée.";
    }
    toast.classList.add("is-visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("is-visible"), 3500);
  });
});
