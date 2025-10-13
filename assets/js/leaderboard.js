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

function createLeaderboardEntry(entry, rank) {
    return `
        <div class="leaderboard-entry" data-rank="${rank}">
            <div class="rank">#${rank}</div>
            <div class="user">${entry.email}</div>
            <div class="points">
                <i class="fas fa-star"></i>
                ${entry.points}
            </div>
            <div class="challenges">${entry.submissions_count}</div>
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
        .map((entry, index) => createLeaderboardEntry(entry, index + 1))
        .join('');
}

// Initial load
document.addEventListener('DOMContentLoaded', updateLeaderboard);

// Refresh every 5 minutes
setInterval(updateLeaderboard, 5 * 60 * 1000);