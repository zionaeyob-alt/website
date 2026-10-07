(() => {
  const root = document.documentElement;
  root.classList.remove("no-js");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Theme toggle (remembered per visitor)
  const stored = (() => { try { return localStorage.getItem("theme"); } catch { return null; } })();
  if (stored) root.dataset.theme = stored;
  document.querySelector(".theme-toggle").addEventListener("click", () => {
    const isDark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = isDark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch {}
  });

  // Nav background on scroll
  const nav = document.querySelector(".nav");
  const onScroll = () => nav.classList.toggle("is-scrolled", window.scrollY > 40);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // Rotating word in hero
  const words = ["innovation", "health research", "AI policy", "technology", "science"];
  const word = document.querySelector(".rotator__word");
  if (word && !reduced) {
    let i = 0;
    setInterval(() => {
      i = (i + 1) % words.length;
      word.classList.add("is-out");
      setTimeout(() => {
        word.textContent = words[i];
        word.classList.remove("is-out");
        word.classList.add("is-in");
        void word.offsetWidth;
        word.classList.remove("is-in");
      }, 450);
    }, 2600);
  }

  // Count-up numbers
  const countUp = (el) => {
    const target = +el.dataset.count;
    if (reduced) { el.textContent = target; return; }
    const start = performance.now(), dur = 1600;
    const tick = (now) => {
      const p = Math.min((now - start) / dur, 1);
      el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  // Reveal on scroll
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      const el = e.target;
      el.classList.add("is-visible");
      el.querySelectorAll("[data-count]").forEach(countUp);
      io.unobserve(el);
    });
  }, { threshold: 0.15, rootMargin: "0px 0px -40px 0px" });

  document.querySelectorAll(".reveal").forEach((el, i) => {
    el.style.transitionDelay = `${Math.min(i % 4, 3) * 80}ms`;
    io.observe(el);
  });

  document.getElementById("year").textContent = new Date().getFullYear();
})();
