const loginModal = document.getElementById('loginModal');
const loginBtn = document.getElementById('login-btn');
const authTabs = document.querySelectorAll('.auth-tab');
const authForms = document.querySelectorAll('.auth-form');
const closeAuthBtn = loginModal.querySelector('.auth-modal-close');

// Open modal
loginBtn.addEventListener('click', () => {
    loginModal.classList.add('active');
    document.body.style.overflow = 'hidden';
});

// Close modal
function closeAuthModal() {
    loginModal.classList.remove('active');
    document.body.style.overflow = '';
}

closeAuthBtn.addEventListener('click', closeAuthModal);
loginModal.addEventListener('click', (e) => {
    if (e.target === loginModal) {
        closeAuthModal();
    }
});

// Tab switching
authTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs and forms
        authTabs.forEach(t => t.classList.remove('active'));
        authForms.forEach(f => f.classList.remove('active'));
        
        // Add active class to clicked tab and corresponding form
        tab.classList.add('active');
        const formId = tab.dataset.tab + 'Form';
        document.getElementById(formId).classList.add('active');
    });
});

// Form submission
document.getElementById('signinForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    // Add your sign in logic here
});

document.getElementById('registerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    // Add your registration logic here
});

// Social auth handlers
document.querySelectorAll('.auth-social-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const provider = btn.classList.contains('github') ? 'github' : 'google';
        // Add your social auth logic here
    });
});