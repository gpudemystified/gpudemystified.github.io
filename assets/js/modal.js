const modal = document.getElementById('challengeModal');
const closeBtn = document.querySelector('.challenge-modal-close');
const runBtn = document.getElementById('runCode');

function openChallenge(challengeId) {
    const challenge = mockChallenges.find(c => c.id === challengeId);
    if (!challenge) return;

    // Update modal content
    document.getElementById('challenge-title').textContent = challenge.title;
    document.getElementById('challenge-description').textContent = challenge.description;
    document.getElementById('challenge-tags').innerHTML = challenge.tags
        .map(tag => `<span class="tag ${tag.toLowerCase()}">${tag}</span>`)
        .join('');

    // Update editor content
    if (editor) {
        editor.setValue(challenge.code);
    }

    // Clear output
    document.getElementById('output').textContent = '';

    // Show modal
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Trigger editor resize
    if (editor) {
        setTimeout(() => editor.layout(), 100);
    }
}

function closeModal() {
    modal.classList.remove('active');
    document.body.style.overflow = '';
}

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