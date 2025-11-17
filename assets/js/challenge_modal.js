const modal = document.getElementById('challengeModal');
const closeBtn = document.querySelector('.challenge-modal-close');
const overlay = modal.querySelector('.challenge-modal-overlay');
const runBtn = document.getElementById('runCode');

let currentChallengeId = null;

// Add this variable at the top with other globals
let hintUsedForChallenge = false;

// Template for response
// {
//   "result": "passed" | "failed",
//   "steps": [
//     {
//       "name": "compilation",
//       "status": "passed" | "failed",
//       "message": "Compilation successful" | "Error message here",
//       "details": "Optional: compiler warnings, line numbers, etc."
//     },
//     {
//       "name": "validation",
//       "status": "passed" | "failed" | "skipped",
//       "message": "All test cases passed" | "Test case 2 failed: expected 42, got 24",
//       "details": "Optional: detailed test results, comparison data"
//     },
//     {
//       "name": "profiling",
//       "status": "passed" | "failed" | "skipped",
//       "message": "Performance metrics collected" | "Profiling failed",
//       "details": "Optional: execution time, memory usage, etc."
//     }
//   ],
//   "user_output": "stdout from user's program",
//   "metadata": {
//     "execution_time_ms": 150,
//     "memory_used_mb": 512,
//     "gpu_used": "NVIDIA RTX 4090"
//   }
// }

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

    populateGPUSelect('gpu-select', true);

    currentChallengeId = challengeId;

    // Update title and points
    document.getElementById('challenge-title').textContent = challenge.title;
    document.getElementById('modal-challenge-points').textContent = `+${challenge.points}`;
    
    // Update tags
    document.getElementById('challenge-tags').innerHTML = challenge.tags
        .map(tag => `<span class="tag ${tag.toLowerCase()}">${tag}</span>`)
        .join('');
    
    // Render markdown description
    const descriptionHtml = marked.parse(challenge.description || '');
    document.getElementById('challenge-description').innerHTML = descriptionHtml;

    // Render math expressions
    if (window.MathJax) {
        MathJax.typesetPromise([document.getElementById('challenge-description')]);
    }

    // Show modal first
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    hintUsedForChallenge = false;

    try {
        // Get profile info from window.userProfile
        const profile = window.userProfile;
        console.log('User profile in modal:', profile);
        
        // Update submissions and hints counters in the meta tab
        const submissionsEl = document.getElementById('modal-submissions-count');
        const hintsEl = document.getElementById('modal-hints-count');
        
        if (profile?.is_pro) {
            submissionsEl.innerHTML = '<i class="fas fa-infinity"></i>';
            hintsEl.innerHTML = '<i class="fas fa-infinity"></i>';
        } else {
            submissionsEl.textContent = profile?.submissions_count ?? '0';
            hintsEl.textContent = profile?.hints_count ?? '0';
        }
        
        let codeToUse = challenge.initial_code;

        // Only load saved progress if user is pro
        if (profile?.is_pro) {
            const savedCode = await window.loadProgress(challengeId);
            if (savedCode.exists) {
                codeToUse = savedCode.code;
            }
        }

        // Recreate editor if it doesn't exist or was disposed
        if (!editor && document.getElementById('monaco-editor')) {
            editor = monaco.editor.create(document.getElementById('monaco-editor'), {
                    value: codeToUse || '// No code available',
                    language: 'cuda',
                    theme: 'vs-light',
                    minimap: { enabled: false },
                    automaticLayout: true
                });

            console.log('Created new Monaco editor');
        } else if (editor) {
            // Editor exists, just update the value
            editor.setValue(codeToUse || '// No code available');
            requestAnimationFrame(() => {
                editor.layout();
                editor.focus();
            });
        }

        // Setup save button with pro-only functionality
        const saveBtn = document.getElementById('saveProgress');
        if (saveBtn) {
            // Remove existing crown icon if any
            const existingIcon = saveBtn.querySelector('.pro-icon');
            if (existingIcon) {
                existingIcon.remove();
            }

            if (!profile?.is_pro) {
                saveBtn.classList.add('disabled');
                saveBtn.title = 'Upgrade to Pro to save your code';

                // Add crown icon at the start of the button
                const proIcon = document.createElement('i');
                proIcon.className = 'fas fa-crown pro-icon';
                saveBtn.insertBefore(proIcon, saveBtn.firstChild);
                
                // Add tooltip functionality
                saveBtn.addEventListener('mouseover', () => {
                    const rect = saveBtn.getBoundingClientRect();
                    const tooltip = document.createElement('div');
                    tooltip.className = 'pro-tooltip';
                    tooltip.textContent = 'Upgrade to Pro to save your code';
                    document.body.appendChild(tooltip);
                    
                    tooltip.style.left = `${rect.left}px`;
                    tooltip.style.top = `${rect.bottom + 5}px`;
                    
                    saveBtn.addEventListener('mouseleave', () => {
                        tooltip.remove();
                    });
                });
            } else {
                saveBtn.classList.remove('disabled');
                saveBtn.title = 'Save your code';
                
                // Only add click handler if user is pro
                saveBtn.addEventListener('click', async () => {
                    try {
                        await window.saveProgress(challengeId, editor.getValue());
                        
                        // Show success feedback
                        saveBtn.classList.add('saved');
                        saveBtn.querySelector('.save-text').textContent = 'Saved!';
                        
                        setTimeout(() => {
                            saveBtn.classList.remove('saved');
                            saveBtn.querySelector('.save-text').textContent = 'Save';
                        }, 2000);
                    } catch (error) {
                        alert('Failed to save progress. Please try again.');
                    }
                });
            }
        }

        // Clear output
        document.getElementById('output').textContent = '';

        // Update hint button state
        await updateHintButtonState();

    } catch (error) {
        console.error('Error loading challenge:', error);
        if (editor) {
            editor.setValue(challenge.initial_code || '// No code available');
        }
    }
}

function closeModal() {
    modal.classList.remove('active');
    document.body.style.overflow = '';
    
    // Dispose of Monaco editor to free resources
    if (editor) {
        editor.dispose();
        editor = null;
        console.log('Disposed Monaco editor');
    }

    // Clear output
    document.getElementById('output').textContent = '';

    // Update challenges grid to reflect any completions
    renderChallenges();
}

// Update event listeners
closeBtn.onclick = closeModal;
modal.querySelector('.challenge-modal-overlay').onclick = closeModal;
document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
        closeModal();
    }
});

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
    const runBtn = document.getElementById('runCode');

    // Disable button immediately
    runBtn.disabled = true;
    runBtn.classList.add('disabled');
    const originalHTML = runBtn.innerHTML;
    runBtn.innerHTML = 'Running... <i class="fas fa-spinner fa-spin"></i>';

    try {
        // Check if user is logged in
        const { data: { session }, error } = await window.supabaseClient.auth.getSession();
        if (!session) {
            OutputFormatter.showLoginRequired(output);
            return;
        }

        // Check submissions before running
        const profile = window.userProfile;
        if (!profile?.is_pro && (!profile?.submissions_count || profile.submissions_count <= 0)) {
            OutputFormatter.showNoSubmissionsError(output);
            return;
        }

        // Format challenge ID
        const formattedId = `challenge_${currentChallengeId}`;

        // Get selected GPU
        const selectedGpu = document.getElementById('gpu-select').value;

        // Create request payload with userId and GPU
        const payload = {
            id: formattedId,
            code: code,
            debug: false,
            user_id: session.user.id
        };

        console.log('Sending request to /run:', payload);

        const response = await fetch(`${getApiUrl()}/run`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${session.access_token}`
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
            
            // Check if it's a submissions limit error from backend
            if (response.status === 429 || (errorText && errorText.includes('submission'))) {
                OutputFormatter.showNoSubmissionsError(output);
                return;
            }
            
            throw new Error(`Server error (${response.status})`);
        }

        const result = await response.json();
        console.log('Server response:', result);
        
        // Use the formatter utility
        const formattedOutput = OutputFormatter.format(result);
        output.innerHTML = formattedOutput;

        // Show completion popup if it's the first completion
        if (result.first_completion) {
            console.log('First completion achieved!');
            const points = document.getElementById('modal-challenge-points').textContent;
            showCompletionPopup(points);
        }

        // Update profile after running code
        await window.updateUserProfile();

        // Update the meta tab counters
        const submissionsEl = document.getElementById('modal-submissions-count');
        
        if (window.userProfile?.is_pro) {
            submissionsEl.innerHTML = '<i class="fas fa-infinity"></i>';
        } else {
            submissionsEl.textContent = window.userProfile?.submissions_count ?? '0';
        }

    } catch (error) {
        console.error('Run code error:', error);
        output.innerHTML = `<span class="error-message">❌ Error: ${OutputFormatter.escapeHtml(error.message)}</span>`;
    } finally {
        // Re-enable button after response (success or error)
        runBtn.disabled = false;
        runBtn.classList.remove('disabled');
        runBtn.innerHTML = originalHTML;
    }
}

function showCompletionPopup(points) {
    const existingPopup = document.querySelector('.completion-popup');
    if (existingPopup) {
        existingPopup.remove();
    }

    const popup = document.createElement('div');
    popup.className = 'completion-popup';
    popup.innerHTML = `
        <i class="fas fa-check-circle"></i>
        <span>Challenge completed ${points} <i class="fas fa-star points-star"></i></span>
    `;

    document.body.appendChild(popup);

    popup.addEventListener('animationend', () => {
        popup.remove();
    });
}

async function handleHintRequest() {
    const hintButton = document.querySelector('.hint-button');
    
    // Check if hint was already used for this challenge
    if (hintUsedForChallenge) {
        return;
    }
    
    // Disable button immediately to prevent multiple clicks
    if (hintButton.disabled) return;
    hintButton.disabled = true;
    hintButton.classList.add('disabled');
    
    try {
        // Get current user session
        const { data: { session }, error: authError } = await window.supabaseClient.auth.getSession();
        
        if (!session) {
            alert('Please login to get hints');
            // Re-enable button if login is required
            hintButton.disabled = false;
            hintButton.classList.remove('disabled');
            return;
        }

        // Construct the hint request URL
        const url = `${getApiUrl()}/hints/challenge_${currentChallengeId}?user_id=${session.user.id}`;
        console.log('Requesting hint:', url);

        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${session.access_token}`
            }
        });
        
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || 'Failed to get hint');
        }

        // Mark hint as used for this challenge
        hintUsedForChallenge = true;

        // Update the challenge description with the hint
        const descriptionEl = document.getElementById('challenge-description');
        const currentDescription = descriptionEl.innerHTML;
        
        // Add hint directly from data.hints
        const hintHtml = marked.parse(data.hints);
        descriptionEl.innerHTML = currentDescription + hintHtml;

        // Update hint button title and keep it disabled
        hintButton.title = 'Hint already used for this challenge';

        // Update profile after using hint
        await window.updateUserProfile();
        
        // Update the meta tab counters with infinity for Pro
        const profile = window.userProfile;
        const hintsEl = document.getElementById('modal-hints-count');
        
        if (profile?.is_pro) {
            hintsEl.innerHTML = '<i class="fas fa-infinity"></i>';
        } else {
            hintsEl.textContent = profile?.hints_count ?? '0';
        }

    } catch (error) {
        console.error('Error getting hint:', error);
        alert('Failed to get hint: ' + error.message);
        // Re-enable button on error
        hintButton.disabled = false;
        hintButton.classList.remove('disabled');
    }
}

// Helper function to escape HTML
function escapeHtml(text) {
    if (!text) return '';
    if (typeof text !== 'string') {
        text = JSON.stringify(text, null, 2);
    }
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
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
});

// Listen for auth state changes
window.supabaseClient.auth.onAuthStateChange((event, session) => {
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