// assets/ui.js — sticky TOC highlight + back to top
(function () {
  var links = Array.prototype.slice.call(document.querySelectorAll('.toc a'));
  var sections = links.map(function (a) {
    return document.querySelector(a.getAttribute('href'));
  }).filter(Boolean);

  function onScroll() {
    var y = window.scrollY + 90;
    var activeIdx = -1;
    sections.forEach(function (sec, i) {
      if (sec.offsetTop <= y) activeIdx = i;
    });
    links.forEach(function (a, i) {
      a.classList.toggle('active', i === activeIdx);
    });
    var btn = document.getElementById('toTop');
    if (btn) btn.classList.toggle('show', window.scrollY > 600);
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  var btn = document.getElementById('toTop');
  if (btn) {
    btn.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  links.forEach(function (a) {
    a.addEventListener('click', function (e) {
      var target = document.querySelector(a.getAttribute('href'));
      if (!target) return;
      e.preventDefault();
      window.scrollTo({ top: target.offsetTop - 56, behavior: 'smooth' });
      history.replaceState(null, '', a.getAttribute('href'));
    });
  });
})();
