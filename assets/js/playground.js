let sourceEditor, assemblyEditor;

require.config({ 
    paths: { 
        vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' 
    }
});

require(['vs/editor/editor.main'], function() {
    // Initialize source code editor
    sourceEditor = monaco.editor.create(document.getElementById('monaco-editor'), {
        value: `__global__ void example(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;
}`,
        language: 'cuda',
        theme: 'vs-light',
        minimap: { enabled: false },
        fontSize: 13,
        scrollBeyondLastLine: false,
        automaticLayout: true
    });

    // Initialize assembly view editor
    assemblyEditor = monaco.editor.create(document.getElementById('assembly-output'), {
        value: '',
        language: 'plaintext',
        theme: 'vs-light',
        readOnly: true,
        minimap: { enabled: false },
        fontSize: 13,
        scrollBeyondLastLine: false,
        lineNumbers: 'on',
        renderLineHighlight: 'all',
        automaticLayout: true,
        wordWrap: 'off'
    });

    const modal = document.getElementById('playgroundModal');
    const closeBtn = modal.querySelector('.challenge-modal-close');
    const compileBtn = document.getElementById('compileBtn');

    // Register CUDA language
    monaco.languages.register({ id: 'cuda' });
    
    // Add CUDA completion items
    monaco.languages.registerCompletionItemProvider('cuda', {
        provideCompletionItems: () => {
            const suggestions = [
                {
                    label: '__global__',
                    kind: monaco.languages.CompletionItemKind.Keyword,
                    insertText: '__global__',
                    detail: 'CUDA global function decorator',
                    documentation: 'Declares a function that runs on the GPU and is callable from the CPU'
                },
                {
                    label: 'threadIdx',
                    kind: monaco.languages.CompletionItemKind.Variable,
                    insertText: 'threadIdx.${1:x}',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Thread index within a block',
                    documentation: 'Access the current thread index (x, y, or z component)'
                },
                {
                    label: 'blockIdx',
                    kind: monaco.languages.CompletionItemKind.Variable,
                    insertText: 'blockIdx.${1:x}',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Block index within the grid',
                    documentation: 'Access the current block index (x, y, or z component)'
                },
                {
                    label: 'cudaMalloc',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'cudaMalloc((void**)&${1:ptr}, ${2:size});',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Allocate memory on the GPU',
                    documentation: 'Allocates size bytes of linear memory on the device'
                },
                {
                    label: '__shared__',
                    kind: monaco.languages.CompletionItemKind.Keyword,
                    insertText: '__shared__ ${1:type} ${2:variable}[${3:size}];',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'CUDA shared memory decorator',
                    documentation: 'Declares a variable in shared memory, accessible by all threads within the same block'
                },
                {
                    label: 'cudaMemcpy',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'cudaMemcpy(${1:dst}, ${2:src}, ${3:size}, ${4:cudaMemcpyHostToDevice});',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Copy memory between host and device',
                    documentation: {
                        value: [
                            '```cuda',
                            'cudaError_t cudaMemcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind)',
                            '```',
                            'Copies data between host and device memory.',
                            '\n\nMemcpy kinds:',
                            '- cudaMemcpyHostToDevice: Host -> Device',
                            '- cudaMemcpyDeviceToHost: Device -> Host',
                            '- cudaMemcpyDeviceToDevice: Device -> Device'
                        ].join('\n')
                    }
                },
                {
                    label: 'blockDim',
                    kind: monaco.languages.CompletionItemKind.Variable,
                    insertText: 'blockDim.${1:x}',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Block dimensions',
                    documentation: 'Number of threads in each dimension of a block'
                },
                {
                    label: 'gridDim',
                    kind: monaco.languages.CompletionItemKind.Variable,
                    insertText: 'gridDim.${1:x}',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Grid dimensions',
                    documentation: 'Number of blocks in each dimension of the grid'
                },
                {
                    label: '__syncthreads',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: '__syncthreads();',
                    detail: 'Synchronize threads in a block',
                    documentation: 'Creates a barrier where all threads in a block must wait before proceeding'
                },
                {
                    label: '__syncwarp',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: '__syncwarp();',
                    detail: 'Synchronize threads in a warp',
                    documentation: 'Creates a barrier where all threads in a warp must wait before proceeding'
                },
                {
                    label: 'atomicAdd',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'atomicAdd(${1:address}, ${2:value})',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Atomic addition',
                    documentation: 'Atomically adds value to the variable at address'
                },
                {
                    label: 'atomicSub',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'atomicSub(${1:address}, ${2:value})',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Atomic subtraction',
                    documentation: 'Atomically subtracts value from the variable at address'
                },
                {
                    label: 'atomicExch',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'atomicExch(${1:address}, ${2:value})',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Atomic exchange',
                    documentation: 'Atomically exchanges value with the value at address'
                },
                {
                    label: 'atomicMin',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'atomicMin(${1:address}, ${2:value})',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Atomic minimum',
                    documentation: 'Atomically computes minimum of value and the value at address'
                },
                {
                    label: 'atomicAnd',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'atomicAnd(${1:address}, ${2:value})',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Atomic AND',
                    documentation: 'Atomically performs bitwise AND of value and the value at address'
                },
                {
                    label: 'atomicCAS',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'atomicCAS(${1:address}, ${2:compare}, ${3:value})',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Atomic Compare-and-Swap',
                    documentation: 'Atomically performs compare-and-swap operation'
                },
                {
                    label: 'cudaFree',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'cudaFree(${1:ptr});',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Free device memory',
                    documentation: 'Frees memory previously allocated by cudaMalloc'
                },
                {
                    label: 'cudaMemset',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'cudaMemset(${1:ptr}, ${2:value}, ${3:size});',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Set device memory',
                    documentation: 'Sets device memory to a value'
                },
                {
                    label: 'cudaMallocManaged',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'cudaMallocManaged((void**)&${1:ptr}, ${2:size});',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Allocate managed memory',
                    documentation: 'Allocates managed memory accessible by both CPU and GPU'
                },
                {
                    label: 'cudaStreamCreate',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'cudaStreamCreate(&${1:stream});',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Create CUDA stream',
                    documentation: 'Creates a new asynchronous stream'
                },
                {
                    label: 'cudaStreamDestroy',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'cudaStreamDestroy(${1:stream});',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Destroy CUDA stream',
                    documentation: 'Destroys and cleans up an asynchronous stream'
                },
                {
                    label: 'cudaStreamSynchronize',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'cudaStreamSynchronize(${1:stream});',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Synchronize CUDA stream',
                    documentation: 'Waits for all operations in the stream to complete'
                },
                {
                    label: 'cudaEventCreate',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'cudaEventCreate(&${1:event});',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Create CUDA event',
                    documentation: 'Creates a new event'
                },
                {
                    label: 'cudaEventRecord',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'cudaEventRecord(${1:event}, ${2:stream});',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Record CUDA event',
                    documentation: 'Records an event in a stream'
                },
                {
                    label: 'cudaEventSynchronize',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'cudaEventSynchronize(${1:event});',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Synchronize CUDA event',
                    documentation: 'Waits for an event to complete'
                },
                {
                    label: 'cudaEventElapsedTime',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'cudaEventElapsedTime(&${1:ms}, ${2:start}, ${3:stop});',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Calculate elapsed time',
                    documentation: 'Computes the elapsed time between two events in milliseconds'
                },
                {
                    label: 'cudaGetDevice',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'cudaGetDevice(&${1:device});',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    detail: 'Get current device',
                    documentation: 'Gets the current CUDA device'
                },
                {
                    label: 'cudaDeviceSynchronize',
                    kind: monaco.languages.CompletionItemKind.Function,
                    insertText: 'cudaDeviceSynchronize();',
                    detail: 'Synchronize device',
                    documentation: 'Waits for all operations on the device to complete'
                }
            ];
            return { suggestions };
        }
    });

    // Configure CUDA syntax highlighting
    monaco.languages.setMonarchTokensProvider('cuda', {
        keywords: [
            '__global__', '__device__', '__host__', '__shared__',
            'threadIdx', 'blockIdx', 'blockDim', 'gridDim',
            'if', 'else', 'for', 'while', 'do', 'return'
        ],
        
        tokenizer: {
            root: [
                [/[a-zA-Z_]\w*/, {
                    cases: {
                        '@keywords': 'keyword',
                        '@default': 'variable'
                    }
                }],
                [/[{}()]/, 'delimiter'],
                [/\/\/.*$/, 'comment'],
                [/\d*\.\d+([eE][\-+]?\d+)?/, 'number.float'],
                [/\d+/, 'number'],
                [/"([^"\\]|\\.)*$/, 'string.invalid'],
                [/"/, { token: 'string.quote', next: '@string' }]
            ],
            
            string: [
                [/[^\\"]+/, 'string'],
                [/"/, { token: 'string.quote', next: '@pop' }]
            ]
        }
    });

    // Add after Monaco initialization but before event listeners
    function createAssemblyMapping(assembly, sourceMap) {
        const assemblyOutput = document.getElementById('assembly-output');
        assemblyOutput.innerHTML = assembly.split('\n').map((line, index) => {
            const lineNum = index + 1;
            return `<div class="assembly-line" data-line="${lineNum}">
                <span class="line-number">${lineNum}</span>
                <span class="assembly-code">${line}</span>
            </div>`;
        }).join('\n');

        // Add source code to assembly line mapping
        editor.onMouseMove(e => {
            const position = e.target.position;
            if (!position) return;

            const sourceLine = position.lineNumber;
            clearHighlights();
            
            if (sourceMap[sourceLine]) {
                sourceMap[sourceLine].forEach(assemblyLine => {
                    const line = document.querySelector(`.assembly-line[data-line="${assemblyLine}"]`);
                    if (line) line.classList.add('highlighted');
                });
            }
        });
    }

    function clearHighlights() {
        document.querySelectorAll('.assembly-line').forEach(line => {
            line.classList.remove('highlighted');
        });
    }

    // Setup event listeners
    closeBtn.addEventListener('click', () => {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    });

    // Update the compile button click handler
    compileBtn.addEventListener('click', async () => {
        const output = document.getElementById('output');
        const code = sourceEditor.getValue();

        try {
            const response = await fetch('http://localhost:8000/compile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    code: code,
                    generate_sass: false,
                    debug: true  // Enable debug info
                })
            });

            if (response.ok) {
                const data = await response.json();
                console.log('PTX output:', data.ptx);
                console.log('Line mapping:', data.line_mapping);
                console.log('Line info available:', data.line_info);

                if (data.success) {
                    assemblyEditor.setValue(data.ptx);
                    
                    if (data.line_mapping && data.line_info) {
                        // Log each source line and its corresponding PTX lines
                        Object.entries(data.line_mapping).forEach(([sourceLine, ptxLines]) => {
                            console.log(`Source line ${sourceLine} maps to PTX lines:`, ptxLines);
                        });
                        setupHighlighting(data.line_mapping);
                    }
                    
                    output.textContent = 'Compilation successful';
                    
                    // Update profile after successful compilation
                    await window.updateUserProfile();
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
        }
    });

    // Add this function to handle the highlighting
    function setupHighlighting(mapping) {
        // Clear any existing decorations
        sourceEditor.deltaDecorations([], []);
        assemblyEditor.deltaDecorations([], []);

        // Store decorations for removal
        let currentSourceDecorations = [];
        let currentAssemblyDecorations = [];

        // Create a reverse mapping from PTX lines to source lines
        const ptxToSourceMap = new Map();
        let currentSourceLine = null;
        
        // Parse the PTX to find .loc directives
        const ptxLines = assemblyEditor.getValue().split('\n');
        ptxLines.forEach((line, index) => {
            const locMatch = line.match(/\.loc\s+1\s+(\d+)\s+\d+/);
            if (locMatch) {
                currentSourceLine = parseInt(locMatch[1]);
            }
            if (currentSourceLine) {
                ptxToSourceMap.set(index + 1, currentSourceLine);
            }
        });

        // Add mouse move handler for source editor
        sourceEditor.onMouseMove(e => {
            if (!e.target.position) return;

            const sourceLine = e.target.position.lineNumber;
            
            // Find all PTX lines that correspond to this source line
            const ptxLines = Array.from(ptxToSourceMap.entries())
                .filter(([_, sLine]) => sLine === sourceLine)
                .map(([pLine, _]) => pLine);

            if (ptxLines.length > 0) {
                // Highlight source line
                currentSourceDecorations = sourceEditor.deltaDecorations(currentSourceDecorations, [{
                    range: new monaco.Range(sourceLine, 1, sourceLine, 1),
                    options: {
                        isWholeLine: true,
                        className: 'highlighted-line'
                    }
                }]);

                // Highlight all corresponding PTX lines
                currentAssemblyDecorations = assemblyEditor.deltaDecorations(
                    currentAssemblyDecorations,
                    ptxLines.map(line => ({
                        range: new monaco.Range(line, 1, line, 1),
                        options: {
                            isWholeLine: true,
                            className: 'highlighted-line'
                        }
                    }))
                );
            } else {
                // Clear decorations if no mapping exists
                currentSourceDecorations = sourceEditor.deltaDecorations(currentSourceDecorations, []);
                currentAssemblyDecorations = assemblyEditor.deltaDecorations(currentAssemblyDecorations, []);
            }
        });

        // Clear highlights when mouse leaves the editor
        sourceEditor.onMouseLeave(() => {
            currentSourceDecorations = sourceEditor.deltaDecorations(currentSourceDecorations, []);
            currentAssemblyDecorations = assemblyEditor.deltaDecorations(currentAssemblyDecorations, []);
        });
    }

    // Handle window resize
    window.addEventListener('resize', () => {
        if (sourceEditor) sourceEditor.layout();
        if (assemblyEditor) assemblyEditor.layout();
    });
});