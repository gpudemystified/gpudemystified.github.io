// Show profile settings
function showProfileSettings() {
    const modal = document.getElementById('profileModal');
    modal.style.display = 'flex';
    
    // Load current profile data
    const profile = window.userProfile;
    
    // Set greeting with username
    const greeting = document.getElementById('profileGreeting');
    if (profile?.username) {
        greeting.textContent = `Hi, ${profile.username}!`;
        document.getElementById('usernameInput').value = profile.username;
    } else {
        greeting.textContent = 'Hi, User!';
    }
    
    // Set account type badge
    const accountBadge = document.getElementById('profileAccountBadge');
    const accountType = document.getElementById('profileAccountType');
    
    if (profile?.is_pro) {
        accountBadge.classList.remove('basic');
        accountBadge.classList.add('pro');
        accountType.textContent = 'PRO';
    } else {
        accountBadge.classList.remove('pro');
        accountBadge.classList.add('basic');
        accountType.textContent = 'Basic';
    }
}

// Save username
async function saveUsername(e) {
    e.preventDefault();
    
    const username = document.getElementById('usernameInput').value.trim();
    const submitBtn = e.target.querySelector('.auth-submit-btn');
    
    // Validate username
    if (username.length < 3) {
        alert('Username must be at least 3 characters');
        return;
    }
    
    if (!/^[a-zA-Z0-9_-]+$/.test(username)) {
        alert('Username can only contain letters, numbers, underscores, and hyphens');
        return;
    }
    
    // Disable button during submission
    submitBtn.disabled = true;
    submitBtn.textContent = 'Saving...';
    
    try {
        const { data: { session } } = await window.supabaseClient.auth.getSession();
        
        const { data, error } = await window.supabaseClient
            .from('users')
            .update({ username: username })
            .eq('user_id', session.user.id);
        
        if (error) {
            if (error.code === '23505') { // Unique constraint violation
                alert('Username already taken. Please choose another.');
            } else {
                throw error;
            }
            return;
        }
        
        // Update local profile
        await window.updateUserProfile();
        
        alert('Username updated successfully!');
        document.getElementById('profileModal').style.display = 'none';
        
    } catch (error) {
        console.error('Error updating username:', error);
        alert('Failed to update username: ' + error.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Save Username';
    }
}

// Handle reset password from profile
async function handleResetPasswordFromProfile() {
    const { data: { session } } = await window.supabaseClient.auth.getSession();
    
    if (!session?.user?.email) {
        alert('Unable to get your email address');
        return;
    }
    
    const confirmReset = confirm(`Send password reset email to ${session.user.email}?`);
    
    if (!confirmReset) return;
    
    try {
        const { error } = await window.supabaseClient.auth.resetPasswordForEmail(
            session.user.email,
            {
                redirectTo: `${window.location.origin}/reset-password.html`,
            }
        );
        
        if (error) throw error;
        
        alert('Password reset email sent! Please check your inbox.');
        document.getElementById('profileModal').style.display = 'none';
        
    } catch (error) {
        console.error('Password reset error:', error);
        alert('Error sending reset email: ' + error.message);
    }
}