// Tab switching functionality
document.addEventListener('DOMContentLoaded', function() {
    const tabs = document.querySelectorAll('.learn-tab');
    const sections = document.querySelectorAll('.learn-section');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;

            // Remove active class from all tabs and sections
            tabs.forEach(t => t.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));

            // Add active class to clicked tab and corresponding section
            tab.classList.add('active');
            document.getElementById(`${targetTab}-section`).classList.add('active');
        });
    });

    // Notes lightbox functionality
    let modalOpen = false;
    const triggers = document.querySelectorAll('.lightbox-trigger');
    const modal = document.getElementById('lightboxModal');
    
    if (modal && triggers.length > 0) {
        const overlay = modal.querySelector('.lb-overlay');
        const imgEl = document.getElementById('lb-image');
        const titleEl = document.getElementById('lb-title');
        const descInner = document.getElementById('lb-desc-inner');
        const closeBtn = modal.querySelector('.lb-close');
        const imgLeftArrow = modal.querySelector('.lb-img-arrow-left');
        const imgRightArrow = modal.querySelector('.lb-img-arrow-right');

        let lastFocused = null;
        let currentNoteIndex = 0;
        let currentImageIndex = 0;
        let notes = Array.from(triggers).map(t => ({
            title: t.dataset.title,
            descriptionUrl: t.dataset.descriptionUrl,
            images: JSON.parse(t.dataset.images),
            noteId: t.dataset.noteId
        }));

        function openModal(trigger){
            lastFocused = document.activeElement;
            currentNoteIndex = Array.from(triggers).indexOf(trigger);
            currentImageIndex = 0;
            showNote(currentNoteIndex, currentImageIndex);
            modalOpen = true;

            modal.setAttribute('aria-hidden', 'false');
            modal.classList.add('open');
            closeBtn.focus();
            modal.style.display = 'flex';
            document.addEventListener('keydown', handleKeyDown);
            document.body.style.overflow = 'hidden';

            const modalBody = modal.querySelector('.lb-body');
            if(modalBody) modalBody.scrollTop = 0;

            const descInner = modal.querySelector('#lb-desc');
            if(descInner) descInner.scrollTop = 0;
        }

        function closeModal(){
            modal.classList.remove('open');
            modalOpen = false;
            modal.style.display = 'none';
            document.removeEventListener('keydown', handleKeyDown);
            document.body.style.overflow = '';
        }

        function showNote(noteIdx, imgIdx){
            const note = notes[noteIdx];
            imgEl.src = note.images[imgIdx];
            imgEl.alt = note.title;
            titleEl.textContent = note.title;

            gtag('event', note.noteId, {
                event_category: 'note',
                event_label: 'Opened note ' + note.noteId
            });
            
            const descUrl = note.descriptionUrl;
            fetch(descUrl)
                .then(res => res.text())
                .then(html => {
                    descInner.innerHTML = html;
                })
                .catch(err => {
                    console.error("Failed to load description:", err);
                    descInner.textContent = "Description not available.";
                });
        }

        function prevImage(){
            const images = notes[currentNoteIndex].images;
            currentImageIndex = (currentImageIndex - 1 + images.length) % images.length;
            showNote(currentNoteIndex, currentImageIndex);
        }
        
        function nextImage(){
            const images = notes[currentNoteIndex].images;
            currentImageIndex = (currentImageIndex + 1) % images.length;
            showNote(currentNoteIndex, currentImageIndex);
        }

        triggers.forEach(t => {
            t.addEventListener('click', () => openModal(t));
            t.addEventListener('keydown', (e) => {
                if(e.key === 'Enter' || e.key === ' ') { 
                    e.preventDefault(); 
                    openModal(t); 
                }
            });
        });

        overlay.addEventListener('click', closeModal);
        closeBtn.addEventListener('click', closeModal);
        imgLeftArrow.addEventListener('click', prevImage);
        imgRightArrow.addEventListener('click', nextImage);

        function handleKeyDown(e){
            if(e.key === 'Escape'){ e.preventDefault(); closeModal(); return; }
            if(e.key === 'ArrowLeft'){ e.preventDefault(); prevImage(); return; }
            if(e.key === 'ArrowRight'){ e.preventDefault(); nextImage(); return; }
            if(e.key === 'Tab'){
                const focusables = modal.querySelectorAll('button, [tabindex]:not([tabindex="-1"])');
                if(!focusables.length) return;
                const first = focusables[0], last = focusables[focusables.length-1];
                if(e.shiftKey && document.activeElement === first){ 
                    e.preventDefault(); 
                    last.focus(); 
                }
                else if(!e.shiftKey && document.activeElement === last){ 
                    e.preventDefault(); 
                    first.focus(); 
                }
            }
        }
    }
});