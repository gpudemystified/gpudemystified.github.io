const modal = document.getElementById('challengeModal');
const closeBtn = document.querySelector('.challenge-modal-close');
const overlay = modal.querySelector('.challenge-modal-overlay');
const runBtn = document.getElementById('runCode');

let currentChallengeId = null;

function setupMarked() {
    marked.use({
        mangle: false,
        headerIds: false,
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, {language: lang}).value;
            }
            return hljs.highlightAuto(code).value;
        }
    });
}

async function openChallenge(challengeId) {
    const challenge = await getChallengeById(challengeId);
    if (!challenge) return;

    currentChallengeId = challengeId;

    // Update title and points
    document.getElementById('challenge-title').textContent = challenge.title;
    document.getElementById('modal-challenge-points').textContent = `+${challenge.points}`;
    
    // Update tags
    document.getElementById('challenge-tags').innerHTML = challenge.tags
        .map(tag => `<span class="tag ${tag.toLowerCase()}">${tag}</span>`)
        .join('');
    
    // Render markdown description using marked.parse()
    const descriptionHtml = marked.parse(challenge.description || '');
    document.getElementById('challenge-description').innerHTML = descriptionHtml;

    // Render math expressions
    if (window.MathJax) {
        MathJax.typesetPromise([document.getElementById('challenge-description')]);
    }

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

async function updateRunButtonState() {
    const runBtn = document.getElementById('runCode');
    const { data: { session }, error } = await window.supabaseClient.auth.getSession();
    console.log ('Update run button state Auth session:', session);
    if (!session) {
        runBtn.disabled = true;
        runBtn.classList.add('disabled');
        runBtn.title = 'Please login to run code';
    } else {
        runBtn.disabled = false;
        runBtn.classList.remove('disabled');
        runBtn.title = 'Run code (Ctrl + Enter)';
    }
}

async function runCode() {
    const output = document.getElementById('output');
    const code = editor.getValue();

    // Check if user is logged in
    const { data: { session }, error } = await window.supabaseClient.auth.getSession();
    if (!session) {
        output.innerText = "Error: Please login to run code";
        return;
    }

    // Format challenge ID
    const formattedId = `challenge_${currentChallengeId}`;

    // Create request payload with userId
    const payload = {
        id: formattedId,
        code: code,
        debug: false,
        user_id: session.user.id
    };

    // Log the request details
    console.log('Sending request to /run:', {
        originalId: currentChallengeId,
        formattedId: formattedId,
        userId: session.user.id,
        codeLength: code.length,
        fullPayload: payload
    });

    try {
        const response = await fetch("http://localhost:8000/run", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Server response:', {
                status: response.status,
                statusText: response.statusText,
                body: errorText
            });
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Server response:', result);
        output.innerText = JSON.stringify(result, null, 2);
    } catch (error) {
        console.error('Run code error:', error);
        output.innerText = "Error: " + error.message;
    }
}

// Add CSS for disabled button
const style = document.createElement('style');
style.textContent = `
    #runCode.disabled {
        opacity: 0.5;
        cursor: not-allowed;
        background-color: #cccccc;
    }
`;
document.head.appendChild(style);

// Update event listeners
document.addEventListener('DOMContentLoaded', () => {
    setupMarked();
    updateRunButtonState();
});

// Listen for auth state changes
window.supabaseClient.auth.onAuthStateChange((event, session) => {
    updateRunButtonState();
});

document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
        closeModal();
    }
});

// Initialize marked when the page loads
document.addEventListener('DOMContentLoaded', setupMarked);

// Add this after other event listeners
runBtn.addEventListener('click', runCode);

// Add keyboard shortcut for running code
document.addEventListener('keydown', async (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && modal.classList.contains('active')) {
        const runBtn = document.getElementById('runCode');
        if (!runBtn.disabled) {
            await runCode();
        }
    }
});