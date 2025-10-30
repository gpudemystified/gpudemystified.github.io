let compilerEditor, assemblyEditor, profileEditor;
let currentDecorations = []; // Store decorations
let currentLineMapping = null; // Store line mapping from compilation

// Add playground ID constants
const COMPILER_EXPLORER_ID = 'playground_999999999';
const RUN_PROFILE_ID = '99999999998';

// Add default code
const defaultCompilerCode = `__global__ void example(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;
}`;

const defaultProfileCode = `#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    // Your code here
    return 0;
}`;

function disableButton(button, loadingText) {
    button.disabled = true;
    button.classList.add('disabled');
    const originalHTML = button.innerHTML;
    button.innerHTML = loadingText;
    return originalHTML;
}

// Re-enable button and restore original content
function enableButton(button, originalHTML) {
    button.disabled = false;
    button.classList.remove('disabled');
    button.innerHTML = originalHTML;
}

// Update modal counts with infinity for Pro users
function updateModalCounts(submissionsElementId, hintsElementId) {
    const profile = window.userProfile;
    
    const submissionsEl = document.getElementById(submissionsElementId);
    const hintsEl = document.getElementById(hintsElementId);
    
    if (!submissionsEl || !hintsEl) {
        console.error('Count elements not found:', submissionsElementId, hintsElementId);
        return;
    }
    
    if (profile?.is_pro) {
        submissionsEl.innerHTML = '<i class="fas fa-infinity"></i>';
        hintsEl.innerHTML = '<i class="fas fa-infinity"></i>';
    } else {
        submissionsEl.textContent = profile?.submissions_count ?? '0';
        hintsEl.textContent = profile?.hints_count ?? '0';
    }
}

async function setupSaveButton(saveButtonId, editorGetter, storageKey) {
    const saveBtn = document.getElementById(saveButtonId);
    if (!saveBtn) {
        console.error(`Save button not found: ${saveButtonId}`);
        return;
    }
    
    const profile = window.userProfile;
    
    // Remove existing crown icon if any
    const existingIcon = saveBtn.querySelector('.pro-icon');
    if (existingIcon) {
        existingIcon.remove();
    }

    // Remove all existing event listeners by cloning
    const newBtn = saveBtn.cloneNode(true);
    saveBtn.parentNode.replaceChild(newBtn, saveBtn);
    const freshBtn = document.getElementById(saveButtonId);

    if (!profile?.is_pro) {
        freshBtn.classList.add('disabled');
        freshBtn.title = 'Upgrade to Pro to save your code';

        // Add crown icon at the start of the button
        const proIcon = document.createElement('i');
        proIcon.className = 'fas fa-crown pro-icon';
        freshBtn.insertBefore(proIcon, freshBtn.firstChild);
        
        // Add tooltip functionality
        freshBtn.addEventListener('mouseover', () => {
            const rect = freshBtn.getBoundingClientRect();
            const tooltip = document.createElement('div');
            tooltip.className = 'pro-tooltip';
            tooltip.textContent = 'Upgrade to Pro to save your code';
            document.body.appendChild(tooltip);
            
            tooltip.style.left = `${rect.left}px`;
            tooltip.style.top = `${rect.bottom + 5}px`;
            
            freshBtn.addEventListener('mouseleave', () => {
                tooltip.remove();
            }, { once: true });
        });
    } else {
        freshBtn.classList.remove('disabled');
        freshBtn.title = 'Save your code';
        
        // Only add click handler if user is pro
        freshBtn.addEventListener('click', async () => {
            try {
                await window.saveProgress(storageKey, editorGetter());
                
                // Show success feedback
                freshBtn.classList.add('saved');
                const saveText = freshBtn.querySelector('.save-text');
                if (saveText) {
                    saveText.textContent = 'Saved!';
                }
                
                setTimeout(() => {
                    freshBtn.classList.remove('saved');
                    if (saveText) {
                        saveText.textContent = 'Save';
                    }
                }, 2000);
            } catch (error) {
                console.error('Save error:', error);
                alert('Failed to save progress. Please try again.');
            }
        });
    }
}

// Load saved code for Pro users
async function loadSavedCode(storageKey, defaultCode) {
    const profile = window.userProfile;
    
    if (profile?.is_pro) {
        const progress = await window.loadProgress(storageKey);
        if (progress.exists) {
            return progress.code;
        }
    }
    
    return defaultCode;
}

// Parse PTX output to build line mapping
function parsePTXLineMapping(ptxOutput) {
    const lines = ptxOutput.split('\n');
    const lineMap = {};
    const reverseLineMap = {}; // PTX line -> Source line
    let currentSourceLine = null;
    
    lines.forEach((line, index) => {
        const ptxLineNumber = index + 1;
        
        // Check if this line contains a .loc directive
        if (line.includes('.loc')) {
            // Parse .loc directive: .loc FileID LineNum Column
            const parts = line.trim().split(/\s+/);
            if (parts.length >= 3) {
                // parts[0] = '.loc', parts[1] = FileID, parts[2] = LineNum
                currentSourceLine = parts[2];
            }
        }
        
        // If we have an active source line, map this PTX line to it
        if (currentSourceLine) {
            // Forward mapping: source -> PTX lines
            if (!lineMap[currentSourceLine]) {
                lineMap[currentSourceLine] = [];
            }
            lineMap[currentSourceLine].push(ptxLineNumber);
            
            // Reverse mapping: PTX line -> source line
            reverseLineMap[ptxLineNumber.toString()] = currentSourceLine;
        }
    });
    
    console.log('Parsed line mapping from PTX:', lineMap);
    console.log('Reverse line mapping (PTX -> Source):', reverseLineMap);
    return { forward: lineMap, reverse: reverseLineMap };
}

// Setup highlighting based on line mapping
function setupHighlighting(ptxOutput) {
    if (!compilerEditor || !assemblyEditor) {
        console.error('Editors not initialized');
        return;
    }
    
    // Parse the PTX output directly to build the line mapping
    const { forward, reverse } = parsePTXLineMapping(ptxOutput);
    
    currentLineMapping = forward;
    const reverseLineMapping = reverse;
    
    console.log('Source -> PTX mapping:', forward);
    console.log('PTX -> Source mapping:', reverse);
    console.log('Source lines available:', Object.keys(forward));
    
    let sourceDecorations = []; // Track source editor decorations
    
    // Source editor mouse move - highlight PTX lines
    compilerEditor.onMouseMove((e) => {
        const position = e.target.position;
        
        // Clear source highlights when hovering over source editor
        sourceDecorations = compilerEditor.deltaDecorations(sourceDecorations, []);
        
        if (!position) {
            // Clear PTX highlights when not hovering over code
            currentDecorations = assemblyEditor.deltaDecorations(currentDecorations, []);
            return;
        }
        
        const sourceLine = position.lineNumber.toString();
        
        // Check if this source line has a mapping
        if (currentLineMapping[sourceLine]) {
            const linesToHighlight = currentLineMapping[sourceLine];
            console.log(`Hovering source line ${sourceLine}, highlighting PTX lines:`, linesToHighlight);
            
            // Create decorations for each assembly line
            const decorations = linesToHighlight.map(lineNum => ({
                range: new monaco.Range(lineNum, 1, lineNum, 1),
                options: {
                    isWholeLine: true,
                    className: 'assembly-line-highlight',
                    glyphMarginClassName: 'assembly-line-glyph'
                }
            }));
            
            // Apply decorations
            currentDecorations = assemblyEditor.deltaDecorations(currentDecorations, decorations);
            
            // Scroll to first highlighted line
            if (linesToHighlight.length > 0) {
                assemblyEditor.revealLineInCenter(linesToHighlight[0]);
            }
        } else {
            // Clear highlights if no mapping for this line
            currentDecorations = assemblyEditor.deltaDecorations(currentDecorations, []);
        }
    });
    
    // Assembly editor mouse move - highlight source lines
    assemblyEditor.onMouseMove((e) => {
        const position = e.target.position;
        
        // Clear PTX highlights when hovering over assembly editor
        currentDecorations = assemblyEditor.deltaDecorations(currentDecorations, []);
        
        if (!position) {
            // Clear source highlights when not hovering over code
            sourceDecorations = compilerEditor.deltaDecorations(sourceDecorations, []);
            return;
        }
        
        const ptxLine = position.lineNumber.toString();
        
        // Check if this PTX line has a mapping to a source line
        if (reverseLineMapping[ptxLine]) {
            const sourceLineToHighlight = reverseLineMapping[ptxLine];
            console.log(`Hovering PTX line ${ptxLine}, highlighting source line:`, sourceLineToHighlight);
            
            // Create decoration for the source line
            const decorations = [{
                range: new monaco.Range(parseInt(sourceLineToHighlight, 10), 1, parseInt(sourceLineToHighlight, 10), 1),
                options: {
                    isWholeLine: true,
                    className: 'source-line-highlight',
                    glyphMarginClassName: 'source-line-glyph'
                }
            }];
            
            // Apply decorations to source editor
            sourceDecorations = compilerEditor.deltaDecorations(sourceDecorations, decorations);
            
            // Scroll to highlighted source line
            compilerEditor.revealLineInCenter(parseInt(sourceLineToHighlight, 10));
        } else {
            // Clear highlights if no mapping for this PTX line
            sourceDecorations = compilerEditor.deltaDecorations(sourceDecorations, []);
        }
    });
    
    console.log('Bidirectional highlighting setup complete');
}

// Initialize Compiler Explorer modal
async function initializeCompilerExplorer() {
    if (!window.monaco) {
        console.error('Monaco not loaded');
        return;
    }
    
    try {
        // Update user profile
        await window.updateUserProfile();
        
        // Clear outputs
        document.getElementById('compiler-output').textContent = '';
        currentDecorations = [];
        currentLineMapping = null;
        
        if (assemblyEditor) {
            assemblyEditor.setValue('// Assembly output will appear here');
        }
        
        // Load saved code or use default
        const codeToUse = await loadSavedCode(COMPILER_EXPLORER_ID, defaultCompilerCode);
        
        // Initialize source editor if not already initialized
        if (!compilerEditor && document.getElementById('compiler-monaco-editor')) {
            compilerEditor = monaco.editor.create(document.getElementById('compiler-monaco-editor'), {
                value: codeToUse,
                language: 'cuda',
                theme: 'vs-dark',
                minimap: { enabled: false },
                automaticLayout: true
            });

            console.log('Initialized compiler editor');
        } else if (compilerEditor) {
            compilerEditor.setValue(codeToUse);
        }

        // Initialize assembly editor if not already initialized
        if (!assemblyEditor && document.getElementById('assembly-output')) {
            assemblyEditor = monaco.editor.create(document.getElementById('assembly-output'), {
                value: '// Assembly output will appear here',
                language: 'cuda',
                theme: 'vs-light',
                readOnly: true,
                minimap: { enabled: false },
                fontSize: 11,
                scrollBeyondLastLine: false,
                lineNumbers: 'on',
                renderLineHighlight: 'all',
                automaticLayout: true,
                wordWrap: 'off'
            });
            
            console.log('Initialized assembly editor');
        }

        // Setup save button
        await setupSaveButton('compiler-saveProgress', () => compilerEditor.getValue(), COMPILER_EXPLORER_ID);

        // Update modal counts
        updateModalCounts('compiler-submissions-count', 'compiler-hints-count');
        
    } catch (error) {
        console.error('Error initializing compiler explorer:', error);
    }
}

// Initialize Run & Profile modal
async function initializeRunProfile() {
    if (!window.monaco) {
        console.error('Monaco not loaded');
        return;
    }
    
    try {
        // Update user profile
        await window.updateUserProfile();
        
        // Clear output
        document.getElementById('profile-output').textContent = '';
        
        // Load saved code or use default
        const codeToUse = await loadSavedCode(RUN_PROFILE_ID, defaultProfileCode);
        
        // Initialize profile editor if not already initialized
        if (!profileEditor && document.getElementById('profile-monaco-editor')) {
            profileEditor = monaco.editor.create(document.getElementById('profile-monaco-editor'), {
               value: codeToUse,
                language: 'cpp',
                theme: 'vs-dark',
                minimap: { enabled: false },
                automaticLayout: true
            });
        } else if (profileEditor) {
            profileEditor.setValue(codeToUse);
        }

        // Setup save button
        await setupSaveButton('profileSaveProgress', () => profileEditor.getValue(), RUN_PROFILE_ID);
        
        // Update modal counts
        updateModalCounts('profile-submissions-count', 'profile-hints-count');
        
    } catch (error) {
        console.error('Error initializing run & profile:', error);
    }
}

// Open Compiler Explorer modal
async function openCompilerExplorer() {
    const modal = document.getElementById('compilerExplorerModal');
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
    
    // Initialize editors and setup
    await initializeCompilerExplorer();
    
    // Layout editors after modal is visible
    setTimeout(() => {
        if (compilerEditor) compilerEditor.layout();
        if (assemblyEditor) assemblyEditor.layout();
    }, 100);
}

// Open Run & Profile modal
async function openRunProfile() {
    const modal = document.getElementById('runProfileModal');
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
    
    // Initialize editor and setup
    await initializeRunProfile();
    
    // Layout editor after modal is visible
    setTimeout(() => {
        if (profileEditor) profileEditor.layout();
    }, 100);
}

// Close modal
function closeModal(modal) {
    modal.classList.remove('active');
    document.body.style.overflow = '';
}

// Compile code (Compiler Explorer)
async function compileCode() {
    const output = document.getElementById('compiler-output');
    const code = compilerEditor.getValue();
    const compileBtn = document.getElementById('compiler-runCode');
    
    const originalHTML = disableButton(compileBtn, '<i class="fas fa-spinner fa-spin"></i> Compiling...');

    console.log('Compiling code:', code);

    try {
        const { data: { session }, error: authError } = await window.supabaseClient.auth.getSession();
        
        if (!session) {
            output.textContent = "Please login to compile code";
            return;
        }

        const response = await fetch('http://localhost:8000/compile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                code: code,
                generate_sass: false,
                debug: true,
                user_id: session.user.id
            })
        });

        if (response.ok) {
            const data = await response.json();
            
            console.log('Compilation successful');
            console.log('PTX output length:', data.ptx?.length);

            if (data.success) {
                assemblyEditor.setValue(data.ptx);
                output.textContent = 'Compilation successful';
                
                // Setup highlighting by parsing the PTX output directly
                setupHighlighting(data.ptx);
                
                await window.updateUserProfile();
                updateModalCounts('compiler-submissions-count', 'compiler-hints-count');
            } else {
                output.textContent = data.error || 'Compilation failed';
            }
        } else {
            const errorData = await response.json();
            output.textContent = `Compilation failed: ${errorData.detail || 'Unknown error'}`;
        }
    } catch (error) {
        console.error('Error during compilation:', error);
        output.textContent = `Error: ${error.message}`;
    } finally {
        enableButton(compileBtn, originalHTML);
    }
}

// Run and profile code
async function runAndProfile() {
    const output = document.getElementById('profile-output');
    const code = profileEditor.getValue();
    const runBtn = document.getElementById('profileRunBtn');
    
    const originalHTML = disableButton(runBtn, '<i class="fas fa-spinner fa-spin"></i> Running...');
    
    try {
        const { data: { session } } = await window.supabaseClient.auth.getSession();
        if (!session) {
            output.textContent = "Error: Please login to run code";
            return;
        }
        
        const gpu = document.getElementById('profile-gpu-select').value;
        
        const response = await fetch("http://localhost:8000/profile", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                code: code,
                user_id: session.user.id,
                gpu: gpu
            })
        });
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const result = await response.json();
        output.textContent = JSON.stringify(result, null, 2);
        
        await window.updateUserProfile();
        updateModalCounts('profile-submissions-count', 'profile-hints-count');
        
    } catch (error) {
        console.error('Profile error:', error);
        output.textContent = "Error: " + error.message;
    } finally {
        enableButton(runBtn, originalHTML);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Get playground items
    const playgroundItems = document.querySelectorAll('.playground-item');
    
    // Add click listeners
    playgroundItems.forEach(item => {
        item.addEventListener('click', () => {
            const playgroundType = item.dataset.playground;
            
            if (playgroundType === 'compiler-explorer') {
                openCompilerExplorer();
            } else if (playgroundType === 'run-profile') {
                openRunProfile();
            }
        });
    });
    
    // Close button listeners
    document.querySelectorAll('.challenge-modal-close').forEach(closeBtn => {
        closeBtn.addEventListener('click', (e) => {
            const modal = e.target.closest('.challenge-modal');
            closeModal(modal);
        });
    });
    
    // Overlay click listeners
    document.querySelectorAll('.challenge-modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', (e) => {
            const modal = e.target.closest('.challenge-modal');
            closeModal(modal);
        });
    });
    
    // Action buttons
    document.getElementById('compiler-runCode')?.addEventListener('click', compileCode);
    document.getElementById('profileRunBtn')?.addEventListener('click', runAndProfile);
    
    // ESC key to close
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (document.getElementById('compilerExplorerModal')?.classList.contains('active')) {
                closeModal(document.getElementById('compilerExplorerModal'));
            }
            if (document.getElementById('runProfileModal')?.classList.contains('active')) {
                closeModal(document.getElementById('runProfileModal'));
            }
        }
    });
});

window.addEventListener('resize', () => {
    if (compilerEditor) {
        compilerEditor.layout();
    }

    if (assemblyEditor) {
        assemblyEditor.layout();
    }

    if (profileEditor) {
        profileEditor.layout();
    }
});