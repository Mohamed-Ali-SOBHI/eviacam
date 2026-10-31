/* Decorative scene only: no camera, face inference or pointer control. */
(() => {
  const root = document.documentElement;
  const stage = document.querySelector(".stage");
  const canvas = document.querySelector("#face-canvas");
  const cursor = document.querySelector(".demo-cursor");
  const toggle = document.querySelector("#motion-toggle");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const context = canvas.getContext("2d");
  let mesh,
    frame = 0,
    previous = 0,
    elapsed = 0;
  let paused = true,
    inView = true,
    pointerX = 0,
    pointerY = 0;
  let width = 620,
    height = 290;

  if ("IntersectionObserver" in window) {
    const reveals = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            reveals.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.08 },
    );
    document
      .querySelectorAll(".reveal")
      .forEach((element) => reveals.observe(element));
    const visibility = new IntersectionObserver((entries) => {
      inView = entries[0].isIntersecting;
      schedule();
    });
    visibility.observe(stage);
  }

  const canMove = () =>
    !paused && !reduced.matches && !document.hidden && inView;
  function schedule() {
    cancelAnimationFrame(frame);
    previous = 0;
    if (canMove()) frame = requestAnimationFrame(tick);
  }

  function draw(time) {
    const phase = time % 10;
    const tracking = phase < 5;
    const blend = tracking ? Math.min(1, phase, 5 - phase) : 0;
    const yaw = (Math.sin(time * 0.65) * 0.38 + pointerX * 0.18) * blend;
    const pitch = (Math.sin(time * 0.43) * 0.1 + pointerY * 0.07) * blend;
    const progress = tracking ? 0 : Math.min(1, (phase - 5) / 3);
    document.querySelector(".dwell-ring").style.strokeDashoffset =
      214 * (1 - progress);
    document.querySelector(".click-check").style.opacity =
      progress === 1 ? 1 : 0;
    cursor.style.opacity = progress === 1 ? 0.2 : 1;
    const status = document.querySelector("#demo-status");
    const message = tracking
      ? "La tête bouge. Le pointeur suit."
      : progress < 1
        ? "Le pointeur reste immobile sur la cible…"
        : "Le clic est déclenché.";
    if (status.textContent !== message) status.textContent = message;
    cursor.style.transform = `translate(${yaw * 105}px, ${pitch * 150}px) rotate(${-yaw * 12}deg)`;
    const scroll = reduced.matches ? 0 : Math.min(1, window.scrollY / 650);
    stage.style.setProperty(
      "--tilt-y",
      `${-10 + Math.sin(time * 0.35) * 5 + pointerX * 5 + scroll * 18}deg`,
    );
    stage.style.setProperty(
      "--tilt-x",
      `${-5 + pointerY * -3 + scroll * 8}deg`,
    );
    stage.style.setProperty(
      "--lift",
      `${Math.sin(time * 0.7) * 4 - scroll * 16}px`,
    );
    if (!mesh || !context) return;
    context.clearRect(0, 0, width, height);
    const cy = Math.cos(yaw),
      sy = Math.sin(yaw),
      cx = Math.cos(pitch),
      sx = Math.sin(pitch);
    const scale = height * 0.035;
    const projected = mesh.vertices.map(([x, y, z]) => {
      z -= 3;
      const a = x * cy + z * sy,
        b = z * cy - x * sy;
      const c = y * cx - b * sx,
        d = y * sx + b * cx;
      const perspective = 34 / (34 - d);
      return [
        width * 0.285 + a * scale * perspective,
        height * 0.49 - c * scale * perspective,
        d,
      ];
    });
    const sorted = mesh.faces
      .map((face) => ({
        face,
        depth: face.reduce((sum, i) => sum + projected[i][2], 0) / 3,
      }))
      .sort((a, b) => a.depth - b.depth);
    context.lineWidth = 0.55;
    sorted.forEach(({ face, depth }) => {
      context.beginPath();
      face.forEach((index, i) => {
        const p = projected[index];
        if (i) context.lineTo(p[0], p[1]);
        else context.moveTo(p[0], p[1]);
      });
      context.closePath();
      context.fillStyle = `rgba(65,95,81,${Math.max(0.025, 0.05 + depth * 0.011)})`;
      context.strokeStyle = `rgba(57,86,71,${Math.max(0.08, 0.18 + depth * 0.025)})`;
      context.fill();
      context.stroke();
    });
    projected.forEach(([x, y, z], i) => {
      if (i % 3) return;
      context.beginPath();
      context.arc(x, y, 0.8, 0, Math.PI * 2);
      context.fillStyle = `rgba(44,71,57,${Math.max(0.15, 0.45 + z * 0.05)})`;
      context.fill();
    });
    [1, 33, 263, 61, 291].forEach((index) => {
      const [x, y] = projected[index];
      context.beginPath();
      context.arc(x, y, 2, 0, Math.PI * 2);
      context.fillStyle = "#325544";
      context.shadowColor = "#53755f";
      context.shadowBlur = 8;
      context.fill();
      context.shadowBlur = 0;
    });
  }

  function tick(now) {
    if (!canMove()) return;
    if (!previous) previous = now;
    const delta = now - previous;
    if (delta >= 32) {
      elapsed += Math.min(delta, 80) / 1000;
      previous = now;
      draw(elapsed);
    }
    frame = requestAnimationFrame(tick);
  }

  function resize() {
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context?.setTransform(ratio, 0, 0, ratio, 0, 0);
    draw(elapsed);
  }

  function preferences() {
    root.classList.toggle(
      "motion-on",
      !reduced.matches && "IntersectionObserver" in window,
    );
    root.classList.toggle("motion-paused", paused || reduced.matches);
    document.body.classList.toggle("reduced-motion", reduced.matches);
    toggle.hidden = false;
    toggle.disabled = reduced.matches;
    toggle.setAttribute("aria-pressed", String(!paused && !reduced.matches));
    toggle.innerHTML = reduced.matches
      ? "Animations réduites selon vos préférences"
      : paused
        ? 'Lancer la démonstration <span aria-hidden="true">▶</span>'
        : 'Mettre en pause <span aria-hidden="true">Ⅱ</span>';
    if (reduced.matches) {
      pointerX = pointerY = 0;
      draw(0);
    }
    schedule();
  }
  toggle.addEventListener("click", () => {
    paused = !paused;
    preferences();
  });
  stage.addEventListener("pointermove", (event) => {
    if (!canMove() || event.pointerType === "touch") return;
    const bounds = stage.getBoundingClientRect();
    pointerX = (event.clientX - bounds.left) / bounds.width - 0.5;
    pointerY = (event.clientY - bounds.top) / bounds.height - 0.5;
  });
  stage.addEventListener("pointerleave", () => {
    if (canMove()) pointerX = pointerY = 0;
  });
  document.addEventListener("visibilitychange", schedule);
  reduced.addEventListener("change", preferences);
  window.addEventListener("resize", resize, { passive: true });
  if (
    /Windows/i.test(navigator.userAgent) &&
    !/Mobile|Android/i.test(navigator.userAgent)
  ) {
    const windows = document.querySelector(".download-link");
    document.querySelector(".hero-download").href = windows.href;
    document.querySelector("#hero-download-label").textContent =
      "Télécharger pour Windows";
    document.querySelector("#hero-download-note").innerHTML =
      'Windows 10 / 11 · 64 bits · Gratuit <a href="#telecharger">Autres versions ↗</a>';
  }
  resize();
  preferences();
  fetch("assets/face-mesh.json")
    .then((response) => {
      if (!response.ok) throw new Error("Mesh unavailable");
      return response.json();
    })
    .then((data) => {
      mesh = data;
      draw(elapsed);
      if (context) root.classList.add("scene-ready");
    })
    .catch(() => {
      /* Static face outline remains usable if the asset fails. */
    });
})();
