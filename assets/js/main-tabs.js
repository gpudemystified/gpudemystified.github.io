document.addEventListener('DOMContentLoaded', function() {
    const tabs = document.querySelectorAll('.main-tab');
    const sections = document.querySelectorAll('.main-section');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;

            // Remove active class from all
            tabs.forEach(t => t.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));

            // Add active class
            tab.classList.add('active');
            document.getElementById(`${targetTab}-section`).classList.add('active');

            // Update URL without reload (optional)
            history.pushState(null, '', `#${targetTab}`);

            // Track tab change
            gtag('event', 'tab_switch', {
                event_category: 'navigation',
                event_label: targetTab
            });
        });
    });

    // Handle direct links (e.g., domain.com#playground)
    function activateTabFromHash() {
        const hash = window.location.hash.slice(1); // Remove #
        if (hash) {
            const targetTab = document.querySelector(`[data-tab="${hash}"]`);
            if (targetTab) {
                targetTab.click();
            }
        }
    }

    // Activate on load
    activateTabFromHash();

    // Handle back/forward buttons
    window.addEventListener('hashchange', activateTabFromHash);
});