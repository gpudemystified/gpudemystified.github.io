<script>
  // Enable scrolling with mouse wheel even when scrollbar is hidden
  document.addEventListener('wheel', function(e) {
    window.scrollBy(0, e.deltaY);
  }, { passive: false });

  // Enable scrolling with touch input even when scrollbar is hidden
  let lastTouchY = null;
  document.addEventListener('touchstart', function(e) {
    if (e.touches.length === 1) {
      lastTouchY = e.touches[0].clientY;
    }
  }, { passive: false });

  document.addEventListener('touchmove', function(e) {
    if (e.touches.length === 1 && lastTouchY !== null) {
      const currentY = e.touches[0].clientY;
      const deltaY = lastTouchY - currentY;
      window.scrollBy(0, deltaY);
      lastTouchY = currentY;
      e.preventDefault();
    }
  }, { passive: false });

  document.addEventListener('touchend', function(e) {
    lastTouchY = null;
  }, { passive: false });
</script>