async function fetchLeaderboard() {
    try {
        const response = await fetch('http://localhost:8000/leaderboard');
        if (!response.ok) throw new Error('Failed to fetch leaderboard');
        return await response.json();
    } catch (error) {
        console.error('Error fetching leaderboard:', error);
        return [];
    }
}

function createLeaderboardEntry(rank, user, points) {
    return `
        <div class="leaderboard-entry">
            <div class="rank">${rank}</div>
            <div class="user">${user}</div>
            <div class="points">${points} <i class="fas fa-star"></i></div>
        </div>
    `;
}

async function updateLeaderboard() {
    const entries = await fetchLeaderboard();
    const container = document.getElementById('leaderboard-entries');
    
    if (!entries.length) {
        container.innerHTML = '<div class="no-entries">No entries yet</div>';
        return;
    }
    
    container.innerHTML = entries
        .slice(0, 50)
        .map((entry, index) => createLeaderboardEntry(index + 1, entry.email, entry.points))
        .join('');
}

// Initial load
document.addEventListener('DOMContentLoaded', updateLeaderboard);

// Refresh every 5 minutes
setInterval(updateLeaderboard, 5 * 60 * 1000);