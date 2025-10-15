async function updateUserProfile() {
    try {
        // Get current user session
        const { data: { session }, error: authError } = await window.supabaseClient.auth.getSession();
        if (!session) return;

        // Fetch updated profile
        const { data: profile, error } = await window.supabaseClient
            .from('profiles')
            .select('is_pro, submissions_count, hints_count, points')
            .eq('id', session.user.id)
            .single();

        if (error) throw error;

        const accountBadge = document.getElementById('account-badge');
        accountBadge.textContent = profile.is_pro ? 'Pro' : 'Basic';
        accountBadge.className = `account-badge ${profile.is_pro ? 'pro' : 'basic'}`;

        // Update UI elements with styled icons for pro accounts
        const submissionsEl = document.getElementById('submissions-count');
        const hintsEl = document.getElementById('hints-count');

        if (profile.is_pro) {
            submissionsEl.innerHTML = '<i class="fas fa-infinity fa-sm"></i>';
            hintsEl.innerHTML = '<i class="fas fa-infinity fa-sm"></i>';
        } else {
            submissionsEl.textContent = profile.submissions_count;
            hintsEl.textContent = profile.hints_count;
        }

        document.getElementById('points-count').textContent = profile.points;

        console.log('Profile updated:', profile);
        return profile;

    } catch (error) {
        console.error('Error updating profile:', error);
        return null;
    }
}

// Make it globally available
window.updateUserProfile = updateUserProfile;