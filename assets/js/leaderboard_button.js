// Update leaderboard button state based on login status
async function updateLeaderboardButton() {
    const leaderboardBtn = document.getElementById('leaderboard-btn');
    
    // If button doesn't exist on this page, return early
    if (!leaderboardBtn) return;
    
    try {
        // Check if user is logged in
        const { data: { session } } = await window.supabaseClient.auth.getSession();
        
        if (!session) {
            // User not logged in - disable and gray out button
            leaderboardBtn.disabled = true;
            leaderboardBtn.style.opacity = '0.5';
            leaderboardBtn.style.cursor = 'not-allowed';
            leaderboardBtn.title = 'Please login to view the leaderboard';
            
            leaderboardBtn.onclick = function(e) {
                e.preventDefault();
                alert('Please login to view the leaderboard');
            };
        } else {
            // User logged in - enable button with normal behavior
            leaderboardBtn.disabled = false;
            leaderboardBtn.style.opacity = '1';
            leaderboardBtn.style.cursor = 'pointer';
            leaderboardBtn.title = 'View Leaderboard';
            
            leaderboardBtn.onclick = function() {
                console.log('Opening leaderboard modal');
                const leaderboardModal = document.getElementById('leaderboardModal');
                if (leaderboardModal) {
                    leaderboardModal.classList.add('active');
                    document.body.style.overflow = 'hidden';
                    // Call updateLeaderboard if it exists
                    if (typeof window.updateLeaderboard === 'function') {
                        window.updateLeaderboard();
                    }
                }
            };
        }
    } catch (error) {
        console.error('Error updating leaderboard button:', error);
        // On error, disable the button
        leaderboardBtn.disabled = true;
        leaderboardBtn.style.opacity = '0.5';
        leaderboardBtn.style.cursor = 'not-allowed';
        leaderboardBtn.title = 'Error checking login status';
    }
}

// Make it globally available
window.updateLeaderboardButton = updateLeaderboardButton;