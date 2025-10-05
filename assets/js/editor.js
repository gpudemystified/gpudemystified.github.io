let editor;

require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' }});

require(['vs/editor/editor.main'], function() {
    // Register CUDA language
    monaco.languages.register({ id: 'cuda' });
    
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

    // Create editor instance
    editor = monaco.editor.create(document.getElementById('monaco-editor'), {
        value: '// CUDA code will appear here',
        language: 'cuda',
        theme: 'vs-light',
        minimap: { enabled: false }
    });
});

// Handle window resize
window.addEventListener('resize', () => {
    if (editor) {
        editor.layout();
    }
});