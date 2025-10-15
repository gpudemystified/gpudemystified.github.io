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

    // First try to load saved progress
    const { data: { session } } = await window.supabaseClient.auth.getSession();
    if (session) {
        const { data, error } = await window.supabaseClient
            .from('challenge_progress')
            .select('code')
            .eq('user_id', session.user.id)
            .eq('challenge_id', challengeId)
            .single();

        // Update editor with saved progress or initial code
        if (editor) {
            if (data?.code) {
                editor.setValue(data.code);
            } else {
                editor.setValue(challenge.initial_code || '// No code available');
            }
            
            requestAnimationFrame(() => {
                editor.layout();
                editor.focus();
            });
        }
    } else {
        // No session, use initial code
        if (editor) {
            editor.setValue(challenge.initial_code || '// No code available');
            requestAnimationFrame(() => {
                editor.layout();
                editor.focus();
            });
        }
    }

    // Show modal
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Clear output
    document.getElementById('output').textContent = '';

    // Update hint button state
    await updateHintButtonState();
    
    const saveBtn = document.getElementById('saveProgress');
    saveBtn.addEventListener('click', () => {
        saveProgress(challengeId, editor.getValue());
    });
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

// Add this function to check hint availability
async function updateHintButtonState() {
    const hintButton = document.querySelector('.hint-button');
    if (!hintButton) return;

    try {
        const { data: { session }, error: authError } = await window.supabaseClient.auth.getSession();
        
        if (!session) {
            hintButton.disabled = true;
            hintButton.classList.add('disabled');
            hintButton.title = 'Please login to use hints';
            return;
        }

        // Fetch user profile to check pro status and hints count
        const { data: profile, error } = await window.supabaseClient
            .from('profiles')
            .select('is_pro, hints_count')
            .eq('id', session.user.id)
            .single();

        if (error) throw error;

        // Disable button if user is not pro and has no hints left
        if (!profile.is_pro && profile.hints_count <= 0) {
            hintButton.disabled = true;
            hintButton.classList.add('disabled');
            hintButton.title = 'No hints available. Upgrade to Pro for unlimited hints!';
        } else {
            hintButton.disabled = false;
            hintButton.classList.remove('disabled');
            hintButton.title = profile.is_pro ? 'Get hint (Pro)' : `Get hint (${profile.hints_count} left)`;
        }

    } catch (error) {
        console.error('Error updating hint button state:', error);
        hintButton.disabled = true;
        hintButton.classList.add('disabled');
        hintButton.title = 'Error checking hint availability';
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

         // Update profile after using hint
        await window.updateUserProfile();
    } catch (error) {
        console.error('Run code error:', error);
        output.innerText = "Error: " + error.message;
    }
}

async function handleHintRequest() {
    const hintButton = document.querySelector('.hint-button');
    
    try {
        // Get current user session
        const { data: { session }, error: authError } = await window.supabaseClient.auth.getSession();
        
        if (!session) {
            alert('Please login to get hints');
            return;
        }

        // Construct the hint request URL
        const url = `http://localhost:8000/hints/challenge_${currentChallengeId}?user_id=${session.user.id}`;
        console.log('Requesting hint:', url);

        const response = await fetch(url);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || 'Failed to get hint');
        }

        // Update the challenge description with the hint
        const descriptionEl = document.getElementById('challenge-description');
        const currentDescription = descriptionEl.innerHTML;
        
        // Add hint directly from data.hints
        const hintHtml = marked.parse(data.hints);
        descriptionEl.innerHTML = currentDescription + hintHtml;

        // Disable the hint button
        hintButton.disabled = true;
        hintButton.classList.add('disabled');
        hintButton.title = 'Hint already used';

        // After successful hint request, update the button state
        await updateHintButtonState();

        // Update profile after using hint
        await window.updateUserProfile();

    } catch (error) {
        console.error('Error getting hint:', error);
        alert('Failed to get hint: ' + error.message);
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

// Add CSS for disabled hint button
const style2 = document.createElement('style');
style2.textContent = `
    .hint-button.disabled {
        opacity: 0.5;
        cursor: not-allowed;
        background-color: #cccccc;
    }
`;
document.head.appendChild(style2);

// Update event listeners
document.addEventListener('DOMContentLoaded', () => {
    setupMarked();
    updateRunButtonState();
});

// Listen for auth state changes
window.supabaseClient.auth.onAuthStateChange((event, session) => {
    updateRunButtonState();
    updateHintButtonState();
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

// Add click handler to hint button
document.addEventListener('DOMContentLoaded', () => {
    const hintButton = document.querySelector('.hint-button');
    if (hintButton) {
        hintButton.addEventListener('click', handleHintRequest);
    }
});

async function saveProgress(challengeId, code) {
    try {
        const { data: { session } } = await window.supabaseClient.auth.getSession();
        if (!session) return;

        const { data, error } = await window.supabaseClient
            .from('challenge_progress')
            .upsert({
                user_id: session.user.id,
                challenge_id: challengeId,
                code: code
            }, {
                onConflict: 'user_id,challenge_id',
                returning: true
            });

        if (error) throw error;

        // Show visual feedback
        const saveBtn = document.getElementById('saveProgress');
        saveBtn.classList.add('saved');
        saveBtn.querySelector('.save-text').textContent = 'Saved!';
        
        setTimeout(() => {
            saveBtn.classList.remove('saved');
            saveBtn.querySelector('.save-text').textContent = 'Save';
        }, 2000);

        return data;
    } catch (error) {
        console.error('Error saving progress:', error);
        alert('Failed to save progress. Please try again.');
    }
}

// Load saved progress for a challenge
async function loadSavedProgress(challengeId) {
    try {
        const { data: { session } } = await window.supabaseClient.auth.getSession();
        if (!session) return;

        const { data, error } = await window.supabaseClient
            .from('challenge_progress')
            .select('code')
            .eq('user_id', session.user.id)
            .eq('challenge_id', challengeId)
            .single();

        if (error) {
            if (error.code === 'PGRST116') {
                // No previous progress found, do nothing
                return;
            }
            throw error;
        }

        // Load the saved code into the editor
        if (editor && data && data.code) {
            editor.setValue(data.code);
            requestAnimationFrame(() => {
                editor.layout();
                editor.focus();
            });
        }

    } catch (error) {
        console.error('Error loading saved progress:', error);
    }
}