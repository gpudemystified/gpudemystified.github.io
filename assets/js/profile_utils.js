async function updateUserProfile() {
    try {
        // Get current user session
        const { data: { session }, error: authError } = await window.supabaseClient.auth.getSession();
        
        // Update leaderboard button state
        if (typeof window.updateLeaderboardButton === 'function') {
            await window.updateLeaderboardButton();
        }
        
        if (!session) {
            window.userProfile = null;
            return;
        }

        // Fetch updated profile
        const { data: profile, error } = await window.supabaseClient
            .from('profiles')
            .select('is_pro, submissions_count, hints_count, points, username')
            .eq('id', session.user.id)
            .single();

        if (error) throw error;

        // Store profile globally
        window.userProfile = profile;
        console.log('Profile stored globally:', window.userProfile);

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
        window.userProfile = null;
        
        // Update leaderboard button on error
        if (typeof window.updateLeaderboardButton === 'function') {
            await window.updateLeaderboardButton();
        }
        
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
            if (confirmBtn) {
                confirmBtn.disabled = true;
                confirmBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>Processing...';
            }

            const { data: { session } } = await window.supabaseClient.auth.getSession();
            
            if (!session) {
                alert('Please login first');
                return;
            }

            const response = await fetch(`${getApiUrl()}/create-checkout-session`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: session.user.id
                })
            });

             console.log('Response status:', response.status);
            
            const responseData = await response.json();
            console.log('Full response data:', responseData);
            
            if (!response.ok) {
                throw new Error(responseData.detail || 'Failed to create checkout session');
            }
            
            // Backend returns 'checkout_url' not 'url'
            const checkoutUrl = responseData.checkout_url;
            
            if (!checkoutUrl) {
                throw new Error('No checkout URL returned from server');
            }
            
            console.log('Redirecting to Stripe checkout:', checkoutUrl);
            
            // Redirect to Stripe Checkout
            window.location.href = checkoutUrl;

        } catch (error) {
            console.error('Error initiating upgrade:', error);
            alert('Failed to start upgrade process. Please try again.');
        }
    };
}

// Make it globally available
window.updateUserProfile = updateUserProfile;