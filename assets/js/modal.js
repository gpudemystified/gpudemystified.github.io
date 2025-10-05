const modal = document.getElementById('challengeModal');
const closeBtn = document.querySelector('.challenge-modal-close');
const overlay = modal.querySelector('.challenge-modal-overlay');
const runBtn = document.getElementById('runCode');

let currentChallengeId = null;  // Add this at the top to track current challenge

async function openChallenge(challengeId) {
    currentChallengeId = challengeId;  // Store the current challenge ID
    console.log("Opening challenge:", challengeId);
    const challenge = await getChallengeById(challengeId);
    if (!challenge) return;

    // Update title and points
    document.getElementById('challenge-title').textContent = challenge.title;
    document.getElementById('modal-challenge-points').textContent = `+${challenge.points}`;
    
    // Update tags and description
    document.getElementById('challenge-tags').innerHTML = challenge.tags
        .map(tag => `<span class="tag ${tag.toLowerCase()}">${tag}</span>`)
        .join('');
    document.getElementById('challenge-description').textContent = challenge.description || challenge.short_description;

    // Update editor content if it exists
    if (editor) {
        editor.setValue(challenge.initial_code || '// No code available');
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

async function runCode() {
    const output = document.getElementById('output');
    const code = editor.getValue();

    try {
        const response = await fetch("http://localhost:8000/run", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                id: currentChallengeId,
                code: code,
                debug: false  // You can add a debug toggle in the UI if needed
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        output.innerText = JSON.stringify(result, null, 2);
    } catch (error) {
        output.innerText = "Error: " + error.message;
        console.error('Run code error:', error);
    }
}

// Update the run button event listener
runBtn.addEventListener('click', runCode);

document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
        closeModal();
    }
});