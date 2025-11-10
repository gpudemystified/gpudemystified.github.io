async function saveProgress(challengeId, code) {
    try {
        const { data: { session } } = await window.supabaseClient.auth.getSession();
        if (!session) {
            console.log('No active session');
            return;
        }

        const response = await fetch(`${getApiUrl()}/progress/save`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${session?.access_token}`
            },
            body: JSON.stringify({
                user_id: session.user.id,
                challenge_id: challengeId,
                code: code
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data;

    } catch (error) {
        console.error('Error saving progress:', error);
        throw error;
    }
}

async function loadProgress(challengeId) {
    try {
        const { data: { session } } = await window.supabaseClient.auth.getSession();
        if (!session) {
            console.log('No active session');
            return { exists: false, code: null };
        }
        
        const url = `${getApiUrl()}/progress/${session.user.id}?challenge_id=${challengeId}`;
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${session.access_token}`
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Loaded progress:', data);
        
        return {
            exists: data.exists,
            code: data.code
        };

    } catch (error) {
        console.error('Error loading progress:', error);
        return { exists: false, code: null };
    }
}

// Make functions globally available
window.saveProgress = saveProgress;
window.loadProgress = loadProgress;