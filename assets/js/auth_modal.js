// Initialize auth state
document.addEventListener('DOMContentLoaded', () => {
    const loginModal = document.getElementById('loginModal');
    const loginBtn = document.getElementById('login-btn');
    const authTabs = document.querySelectorAll('.auth-tab');
    const authForms = document.querySelectorAll('.auth-form');
    const closeAuthBtn = loginModal.querySelector('.auth-modal-close');

    // Define openAuthModal function
    function openAuthModal() {
        loginModal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    // Open modal
    loginBtn.addEventListener('click', openAuthModal);

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

    // Sign in with email/password
    document.getElementById('signinForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = e.target.querySelector('input[type="email"]').value;
        const password = e.target.querySelector('input[type="password"]').value;

        try {
            const { data, error } = await window.supabaseClient.auth.signInWithPassword({
                email,
                password
            });

            if (error) throw error;

            const { data: { session }, error2 } = await window.supabaseClient.auth.getSession();
            console.log('Logged in user:', data.user, session);

            // Success - user is signed in
            closeAuthModal();
            updateUIForAuthenticatedUser(data.user);
        } catch (error) {
            alert('Error signing in: ' + error.message);
        }
    });

    // Register with email/password
    document.getElementById('registerForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = e.target.querySelector('input[type="email"]').value;
        const password = e.target.querySelector('input[type="password"]').value;
        const confirmPassword = e.target.querySelector('input[placeholder="Confirm Password"]').value;

        if (password !== confirmPassword) {
            alert('Passwords do not match');
            return;
        }

        try {
            const { data, error } = await window.supabaseClient.auth.signUp({
                email,
                password
            });

            if (error) throw error;

            alert('Registration successful! Please check your email for confirmation.');
            closeAuthModal();
        } catch (error) {
            alert('Error registering: ' + error.message);
        }
    });

    // Update the social auth handler
    document.querySelectorAll('.auth-social-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const provider = btn.classList.contains('github') ? 'github' : 'google';
            
            try {
                const { data, error } = await window.supabaseClient.auth.signInWithOAuth({
                    provider,
                    options: {
                        redirectTo: `${window.location.origin}/auth/callback`,
                        queryParams: {
                            access_type: 'offline',
                            prompt: 'consent',
                        }
                    }
                });

                if (error) throw error;
                
                // Handle successful redirect
                if (data) {
                    console.log('OAuth login successful:', data);
                    updateUIForAuthenticatedUser(data.user);
                }
            } catch (error) {
                console.error(`Error signing in with ${provider}:`, error);
                alert(`Error signing in with ${provider}: ${error.message}`);
            }
        });
    });

    // Check for existing session
    async function checkAuthSession() {
        const { data: { session }, error } = await window.supabaseClient.auth.getSession();
        
        console.log('Current session:', session);
        if (session) {
            updateUIForAuthenticatedUser(session.user);
        }
    }

    // Update UI when user is authenticated
    function updateUIForAuthenticatedUser(user) {
        const loginBtn = document.getElementById('login-btn');
        loginBtn.textContent = 'Logout';
        loginBtn.removeEventListener('click', openAuthModal);
        loginBtn.addEventListener('click', handleLogout);
    }

    // Handle logout
    async function handleLogout() {
        try {
            const { error } = await window.supabaseClient.auth.signOut();
            if (error) throw error;
            
            // Reset UI
            const loginBtn = document.getElementById('login-btn');
            loginBtn.textContent = 'Login';
            loginBtn.removeEventListener('click', handleLogout);
            loginBtn.addEventListener('click', openAuthModal);
            
        } catch (error) {
            alert('Error signing out: ' + error.message);
        }
    }

    // Add auth state change listener
    window.supabaseClient.auth.onAuthStateChange((event, session) => {
        console.log('Auth state changed:', event, session);
        if (event === 'SIGNED_IN') {
            updateUIForAuthenticatedUser(session.user);
        }
    });

    // Initialize auth session check
    checkAuthSession();
});