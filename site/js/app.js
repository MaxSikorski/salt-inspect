/**
 * app.js — Core functionality: theme, nav, animated counters, contact form
 */
(function () {
  'use strict';

  // ---- Theme Toggle ----
  const themeToggle = document.getElementById('themeToggle');
  const html = document.documentElement;

  function setTheme(theme) {
    html.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }

  // Initialize from saved preference or system
  const saved = localStorage.getItem('theme');
  if (saved) {
    setTheme(saved);
  } else if (window.matchMedia('(prefers-color-scheme: light)').matches) {
    setTheme('light');
  }

  themeToggle.addEventListener('click', () => {
    setTheme(html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  });

  // ---- Navigation ----
  const navToggle = document.querySelector('.nav-toggle');
  const navMenu = document.querySelector('.nav-menu');
  const navBackdrop = document.querySelector('.nav-backdrop');

  function closeNav() {
    navToggle.classList.remove('active');
    navToggle.setAttribute('aria-expanded', 'false');
    navMenu.classList.remove('open');
    navBackdrop.classList.remove('open');
  }

  navToggle.addEventListener('click', () => {
    const isOpen = navMenu.classList.contains('open');
    if (isOpen) {
      closeNav();
    } else {
      navToggle.classList.add('active');
      navToggle.setAttribute('aria-expanded', 'true');
      navMenu.classList.add('open');
      navBackdrop.classList.add('open');
    }
  });

  navBackdrop.addEventListener('click', closeNav);

  navMenu.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', closeNav);
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeNav();
  });

  // ---- Animated Counters ----
  function animateCounters() {
    document.querySelectorAll('.stat-number[data-target]').forEach((el) => {
      const target = parseFloat(el.dataset.target);
      const prefix = el.dataset.prefix || '';
      const suffix = el.dataset.suffix || '';
      const duration = 1500;
      const start = performance.now();

      function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const value = Math.round(target * eased);
        el.textContent = prefix + value + suffix;

        if (progress < 1) {
          requestAnimationFrame(update);
        }
      }

      requestAnimationFrame(update);
    });
  }

  // Trigger counters when hero is visible
  const heroSection = document.getElementById('hero');
  if (heroSection) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            animateCounters();
            observer.disconnect();
          }
        });
      },
      { threshold: 0.3 }
    );
    observer.observe(heroSection);
  }

  // ---- Fade-in on Scroll ----
  const fadeObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
  );

  document.querySelectorAll('.fade-in').forEach((el) => {
    fadeObserver.observe(el);
  });

  // ---- Demo Tabs ----
  const demoTabs = document.querySelectorAll('.demo-tab');
  const tabContents = {
    samples: document.getElementById('samplesTab'),
    upload: document.getElementById('uploadTab'),
    camera: document.getElementById('cameraTab'),
  };
  const categoryFilter = document.getElementById('categoryFilter');

  demoTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;

      demoTabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');

      Object.values(tabContents).forEach((c) => c.classList.remove('active'));
      if (tabContents[target]) tabContents[target].classList.add('active');

      // Show category filter only for samples tab
      if (categoryFilter) {
        categoryFilter.style.display = target === 'samples' ? 'block' : 'none';
      }
    });
  });

  // ---- Contact Form ----
  document.querySelectorAll('.contact-form').forEach((form) => {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = form.querySelector('button[type="submit"]');
      const originalText = submitBtn.textContent;
      submitBtn.disabled = true;
      submitBtn.textContent = 'Sending...';

      const prevError = form.querySelector('.contact-error');
      if (prevError) prevError.remove();

      try {
        const resp = await fetch(form.action, {
          method: 'POST',
          body: new FormData(form),
          headers: { Accept: 'application/json' },
        });

        if (resp.ok) {
          const wrapper = form.parentElement;
          form.style.display = 'none';
          const success = document.createElement('div');
          success.className = 'contact-success';
          success.innerHTML = `
            <div class="success-icon">
              <svg viewBox="0 0 24 24"><path d="M5 13l4 4L19 7"/></svg>
            </div>
            <h3>Sprint request received.</h3>
            <p>We'll respond within 4 hours with next steps.</p>
          `;
          wrapper.appendChild(success);
        } else {
          throw new Error('Server returned ' + resp.status);
        }
      } catch (err) {
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
        const errorEl = document.createElement('div');
        errorEl.className = 'contact-error';
        errorEl.textContent =
          'Something went wrong. Please try again or email maxwell.sikorski@gmail.com directly.';
        form.appendChild(errorEl);
      }
    });
  });

  // ---- Emit section-visible events for lazy loading ----
  const sectionObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          window.dispatchEvent(
            new CustomEvent('section-visible', {
              detail: { id: entry.target.id },
            })
          );
        }
      });
    },
    { threshold: 0.1 }
  );

  document.querySelectorAll('section[id]').forEach((section) => {
    sectionObserver.observe(section);
  });
})();
