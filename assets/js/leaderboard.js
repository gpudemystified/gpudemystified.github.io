async function fetchLeaderboard() {
    try {
        const response = await fetch(`${getApiUrl()}/leaderboard`);
        if (!response.ok) throw new Error('Failed to fetch leaderboard');
        return await response.json();
    } catch (error) {
        console.error('Error fetching leaderboard:', error);
        return [];
    }
}

function createLeaderboardEntry(rank, user, points, isCurrentUser = false) {
    const highlightClass = isCurrentUser ? 'current-user' : '';
    const crown = isCurrentUser ? '<i class="fas fa-user-circle"></i>' : '';
    
    return `
        <div class="leaderboard-entry ${highlightClass}">
            <div class="rank">${rank}</div>
            <div class="user">${crown} ${user}</div>
            <div class="points">${points} <i class="fas fa-star"></i></div>
        </div>
    `;
}

async function updateLeaderboard() {
    console.log("Updating leaderboard...");
    const entries = await fetchLeaderboard();
    const container = document.getElementById('leaderboard-entries');
    
    if (!entries.length) {
        container.innerHTML = '<div class="no-entries">No entries yet</div>';
        return;
    }
    
    // Get current user's username
    const currentUsername = window.userProfile?.username || null;
    
    container.innerHTML = entries
        .slice(0, 50)
        .map((entry, index) => {
            const isCurrentUser = currentUsername && entry.username === currentUsername;
            
            return createLeaderboardEntry(index + 1, entry.username, entry.points, isCurrentUser);
        })
        .join('');
}
