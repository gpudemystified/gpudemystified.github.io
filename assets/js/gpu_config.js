// Available GPUs - simple list
const availableGPUs = [
    { id: 'rtx_4090', name: 'NVIDIA RTX 4090' }
    // Add more GPUs here as they become available:
    // { id: 'a100', name: 'NVIDIA A100' },
    // { id: 'mi250', name: 'AMD MI250' },
];

// Populate GPU dropdown
function populateGPUSelect(selectId, disabled = false) {
    const select = document.getElementById(selectId);
    if (!select) return;
    
    select.innerHTML = availableGPUs.map(gpu => 
        `<option value="${gpu.id}">${gpu.name}</option>`
    ).join('');
    
    // Disable or enable the dropdown
    if (disabled) {
        select.disabled = true;
        select.style.opacity = '0.5';
        select.style.cursor = 'not-allowed';
    } else {
        select.disabled = false;
        select.style.opacity = '1';
        select.style.cursor = 'pointer';
    }
}

// Make it globally available
window.availableGPUs = availableGPUs;
window.populateGPUSelect = populateGPUSelect;