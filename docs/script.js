const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const revealElements = document.querySelectorAll(".reveal");
if (reducedMotion || !("IntersectionObserver" in window)) {
  revealElements.forEach((element) => element.classList.add("is-visible"));
} else {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  revealElements.forEach((element) => observer.observe(element));
}

const cameraFeed = document.querySelector("[data-camera-feed]");
const demoPointer = document.querySelector("[data-demo-pointer]");
if (cameraFeed && demoPointer && !reducedMotion) {
  cameraFeed.addEventListener("pointermove", (event) => {
    const bounds = cameraFeed.getBoundingClientRect();
    const x = ((event.clientX - bounds.left) / bounds.width - 0.72) * 34;
    const y = ((event.clientY - bounds.top) / bounds.height - 0.62) * 34;
    demoPointer.style.transform = `translate(${x}px, ${y}px)`;
  });

  cameraFeed.addEventListener("pointerleave", () => {
    demoPointer.style.transform = "translate(0, 0)";
  });
}

const toast = document.querySelector("[data-copy-toast]");
let toastTimer;

document.querySelectorAll("[data-copy-hash]").forEach((button) => {
  button.addEventListener("click", async () => {
    const hash = button.dataset.copyHash;
    try {
      await navigator.clipboard.writeText(hash);
      button.textContent = "Copié";
      toast?.classList.add("is-visible");
      window.clearTimeout(toastTimer);
      toastTimer = window.setTimeout(() => {
        toast?.classList.remove("is-visible");
        button.textContent = "Copier";
      }, 1800);
    } catch {
      button.textContent = "Sélectionnez le hash";
    }
  });
});
