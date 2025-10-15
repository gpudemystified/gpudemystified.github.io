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
        
        if (profile.is_pro) {
            accountBadge.innerHTML = `<i class="fas fa-crown"></i> Pro`;
            accountBadge.className = 'account-badge pro';
            accountBadge.onclick = null;
        } else {
            accountBadge.innerHTML = `<i class="fas fa-crown"></i> Upgrade to Pro`;
            accountBadge.className = 'account-badge basic';
            accountBadge.onclick = handleUpgrade;
        }

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

async function handleUpgrade() {
    const proModal = document.getElementById('proModal');
    const closeBtn = proModal.querySelector('.pro-modal-close');
    const confirmBtn = document.getElementById('confirmUpgrade');
    const overlay = proModal.querySelector('.pro-modal-overlay');

    // Show modal
    proModal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Handle close actions
    const closeModal = () => {
        proModal.classList.remove('active');
        document.body.style.overflow = '';
    };

    closeBtn.onclick = closeModal;
    overlay.onclick = closeModal;

    // Handle upgrade confirmation
    confirmBtn.onclick = async () => {
        try {
            const { data: { session } } = await window.supabaseClient.auth.getSession();
            
            if (!session) {
                alert('Please login first');
                return;
            }

            const response = await fetch('http://localhost:8000/create-checkout-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: session.user.id,
                    price_id: 'your_stripe_price_id'
                })
            });

            const { url } = await response.json();
            window.location.href = url;

        } catch (error) {
            console.error('Error initiating upgrade:', error);
            alert('Failed to start upgrade process. Please try again.');
        }
    };
}

// Make it globally available
window.updateUserProfile = updateUserProfile;