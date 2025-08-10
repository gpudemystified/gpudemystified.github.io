(function () {
  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function initCardCarousels() {
    const carousels = document.querySelectorAll('.notes-grid .carousel');
    carousels.forEach((carousel) => {
      const track = carousel.querySelector('.carousel-track');
      const slides = carousel.querySelectorAll('.carousel-slide');
      const prev = carousel.querySelector('.carousel-control.prev');
      const next = carousel.querySelector('.carousel-control.next');
      let index = 0;

      function update() {
        const offset = -index * 100;
        track.style.transform = `translateX(${offset}%)`;
      }

      prev?.addEventListener('click', (e) => {
        e.stopPropagation();
        index = clamp(index - 1, 0, slides.length - 1);
        update();
      });

      next?.addEventListener('click', (e) => {
        e.stopPropagation();
        index = clamp(index + 1, 0, slides.length - 1);
        update();
      });

      // Open modal when clicking anywhere on the carousel area (except buttons)
      carousel.addEventListener('click', (e) => {
        if ((e.target instanceof Element) && e.target.closest('.carousel-control')) return;
        const imagesJson = carousel.getAttribute('data-images') || '[]';
        const title = carousel.getAttribute('data-title') || '';
        const text = carousel.getAttribute('data-text') || '';
        let images = [];
        try { images = JSON.parse(imagesJson); } catch {}
        openModal({ images, title, text, startIndex: index });
      });

      update();
    });
  }

  // Modal state
  let modalIndex = 0;
  let modalImages = [];

  const modal = document.getElementById('notesModal');
  const modalTrack = document.getElementById('modalCarouselTrack');
  const modalPrev = document.getElementById('modalPrev');
  const modalNext = document.getElementById('modalNext');
  const modalClose = document.getElementById('modalClose');
  const modalTitle = document.getElementById('modalTitle');
  const modalText = document.getElementById('modalText');

  function renderModalImages(images) {
    modalTrack.innerHTML = '';
    images.forEach((src, i) => {
      const img = document.createElement('img');
      img.className = 'carousel-slide';
      img.src = src;
      img.alt = `${modalTitle.textContent || 'image'} ${i + 1}`;
      modalTrack.appendChild(img);
    });
  }

  function updateModal() {
    const offset = -modalIndex * 100;
    modalTrack.style.transform = `translateX(${offset}%)`;
  }

  function openModal({ images, title, text, startIndex = 0 }) {
    modalImages = images || [];
    modalIndex = clamp(startIndex, 0, Math.max(0, modalImages.length - 1));
    renderModalImages(modalImages);
    modalTitle.textContent = title || '';
    modalText.textContent = text || '';
    modal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    updateModal();
  }

  function closeModal() {
    modal.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
  }

  modalPrev?.addEventListener('click', () => {
    modalIndex = clamp(modalIndex - 1, 0, modalImages.length - 1);
    updateModal();
  });

  modalNext?.addEventListener('click', () => {
    modalIndex = clamp(modalIndex + 1, 0, modalImages.length - 1);
    updateModal();
  });

  modalClose?.addEventListener('click', closeModal);

  modal?.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
  });

  document.addEventListener('keydown', (e) => {
    if (modal.getAttribute('aria-hidden') === 'true') return;
    if (e.key === 'Escape') closeModal();
    if (e.key === 'ArrowLeft') {
      modalIndex = clamp(modalIndex - 1, 0, modalImages.length - 1);
      updateModal();
    }
    if (e.key === 'ArrowRight') {
      modalIndex = clamp(modalIndex + 1, 0, modalImages.length - 1);
      updateModal();
    }
  });

  document.addEventListener('DOMContentLoaded', () => {
    initCardCarousels();
  });
})();