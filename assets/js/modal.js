const modal = document.getElementById('challengeModal');
const closeBtn = document.querySelector('.challenge-modal-close');
const overlay = modal.querySelector('.challenge-modal-overlay');
const runBtn = document.getElementById('runCode');

function openChallenge(challengeId) {
    const challenge = mockChallenges.find(c => c.id === challengeId);
    if (!challenge) return;

    // Update title and points
    document.getElementById('challenge-title').textContent = challenge.title;
    document.getElementById('modal-challenge-points').textContent = `+${challenge.points}`;
    
    // Update tags and description
    document.getElementById('challenge-tags').innerHTML = challenge.tags
        .map(tag => `<span class="tag ${tag.toLowerCase()}">${tag}</span>`)
        .join('');
    document.getElementById('challenge-description').textContent = challenge.description;

    // Update editor content if it exists
    if (editor) {
        editor.setValue(challenge.code);
        requestAnimationFrame(() => {
            editor.layout();
            editor.focus();
        });
    }

    // Show modal
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Clear output
    document.getElementById('output').textContent = '';
}

function closeModal() {
    modal.classList.remove('active');
    document.body.style.overflow = '';
}

overlay.addEventListener('click', closeModal);

closeBtn.addEventListener('click', closeModal);
modal.addEventListener('click', e => {
    if (e.target === modal) closeModal();
});

runBtn.addEventListener('click', () => {
    const output = document.getElementById('output');
    output.textContent = '⚠️ Code execution is not implemented in this demo';
});

document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
        closeModal();
    }
});