// API Configuration
// Set DEBUG to true for local development, false for production
const DEBUG = false;

const API_CONFIG = {
    LOCAL: 'http://localhost:8000',
    PRODUCTION: 'https://api.gpudemystified.com'
};

// Get the current API base URL
function getApiUrl() {
    return DEBUG ? API_CONFIG.LOCAL : API_CONFIG.PRODUCTION;
}

// Make it globally available
window.API_CONFIG = API_CONFIG;
window.DEBUG = DEBUG;
window.getApiUrl = getApiUrl;