const mockChallenges = [
    {
        id: "vector_addition",
        title: "Vector Addition",
        short_description: "Add two vectors element-wise on the GPU.",
        tags: ["CUDA", "Beginner"],
        points: 10,
        code: "#include <cuda_runtime.h> \n\n__global__ void vectorAdd(float *A, float *B, float *C, int N) {\n    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < N) C[i] = A[i] + B[i];\n}"
    },
    {
        id: "matrix_multiplication",
        title: "Matrix Multiplication",
        short_description: "Multiply two matrices using shared memory optimization.",
        tags: ["CUDA", "Intermediate"],
        points: 20,
        code: "__global__ void matMul(float *A, float *B, float *C, int N) {\n    int row = blockIdx.y * blockDim.y + threadIdx.y;\n    int col = blockIdx.x * blockDim.x + threadIdx.x;\n    if (row < N && col < N) {\n        float sum = 0;\n        for (int k = 0; k < N; ++k)\n            sum += A[row * N + k] * B[k * N + col];\n        C[row * N + col] = sum;\n    }\n}"
    },
    {
        id: "image_convolution",
        title: "Image Convolution",
        short_description: "Implement a 2D image convolution kernel.",
        tags: ["CUDA", "Advanced", "Image Processing"],
        points: 30,
        code: "__global__ void conv2d(float *input, float *kernel, float *output, int width, int height, int ksize) {\n    int x = blockIdx.x * blockDim.x + threadIdx.x;\n    int y = blockIdx.y * blockDim.y + threadIdx.y;\n    if (x >= width || y >= height) return;\n    float sum = 0;\n    for (int ky = 0; ky < ksize; ky++) {\n        for (int kx = 0; kx < ksize; kx++) {\n            int ix = x + kx - ksize / 2;\n            int iy = y + ky - ksize / 2;\n            if (ix >= 0 && iy >= 0 && ix < width && iy < height)\n                sum += input[iy * width + ix] * kernel[ky * ksize + kx];\n        }\n    }\n    output[y * width + x] = sum;\n}"
    }
];

let cachedChallenges = null;
let completedChallenges = new Set();

async function getChallenges() {
    if (cachedChallenges) {
        return cachedChallenges;
    }

    try {
        const { data, error } = await window.supabaseClient
            .from('challenges')
            .select('*');

        if (error) throw error;

        console.log('Fetched challenges from Supabase:', data);
        cachedChallenges = data;
        return data;
    } catch (error) {
        console.warn('Failed to fetch challenges from Supabase, using mock data:', error);
        cachedChallenges = mockChallenges;
        return mockChallenges;
    }
}

function sortChallenges(challenges, sortBy) {
    console.log(`Sorting challenges by: ${sortBy}`);
    if (sortBy === 'default') {
        return [...challenges].sort((a, b) => {
            // Handle numeric IDs
            return (a.id || 0) - (b.id || 0);
        });
    }

    return [...challenges].sort((a, b) => {
        switch (sortBy) {
            case 'name':
                return String(a.title || '').localeCompare(String(b.title || ''));
            case 'points':
                return (b.points || 0) - (a.points || 0);
            case 'difficulty':
                const difficultyOrder = { 'Beginner': 1, 'Intermediate': 2, 'Advanced': 3 };
                const aDifficulty = (a.tags || []).find(tag => difficultyOrder[tag]) || 'Beginner';
                const bDifficulty = (b.tags || []).find(tag => difficultyOrder[tag]) || 'Beginner';
                return difficultyOrder[aDifficulty] - difficultyOrder[bDifficulty];
            default:
                // Use numeric comparison for IDs
                return (a.id || 0) - (b.id || 0);
        }
    });
}

async function renderChallenges() {
    const grid = document.getElementById('challenges-grid');
    const sortSelect = document.getElementById('sortSelect');
    
    try {
        // Load both challenges and completion status in parallel
        const [challenges, completedStatus] = await Promise.all([
            getChallenges(),
            loadCompletedChallengesData()  // New function to get just the data
        ]);

        const sortedChallenges = sortChallenges(challenges, sortSelect.value);
        
        grid.innerHTML = sortedChallenges.map(challenge => {
            const isCompleted = completedChallenges.has(`challenge_${challenge.id}`);
            return `
                <div class="challenge-card ${isCompleted ? 'completed' : ''}" data-id="${challenge.id}" 
                     onclick="openChallenge('${challenge.id}')">
                    <div class="challenge-complete">
                        ${isCompleted ? '<i class="fas fa-check-circle"></i>' : ''}
                    </div>
                    <div class="challenge-points">
                        <i class="fas fa-star"></i>
                        +${challenge.points}
                    </div>
                    <h3 class="challenge-title">${challenge.title}</h3>
                    <p>${challenge.short_description}</p>
                    <div class="challenge-tags">
                        ${challenge.tags.map(tag => `
                            <span class="tag ${tag.toLowerCase()}">${tag}</span>
                        `).join('')}
                    </div>
                </div>
            `;
        }).join('');

    } catch (error) {
        console.error('Error rendering challenges:', error);
        grid.innerHTML = '<p>Error loading challenges</p>';
    }
}

// New function to fetch completed challenges data
async function loadCompletedChallengesData() {
    try {
        const { data: { session } } = await window.supabaseClient.auth.getSession();
        if (!session) return new Set();

        const response = await fetch(`http://localhost:8000/challenges/completed/${session.user.id}`);
        if (!response.ok) throw new Error('Failed to fetch completed challenges');

        const completed = await response.json();
        completedChallenges = new Set(completed);
        return completedChallenges;

    } catch (error) {
        console.error('Error loading completed challenges:', error);
        return new Set();
    }
}

async function getChallengeById(challengeId) {
    try {
        const { data, error } = await window.supabaseClient
            .from('challenges')
            .select('*')
            .eq('id', challengeId)
            .single();

        if (error) throw error;
        return data;
    } catch (error) {
        console.warn('Failed to fetch challenge from Supabase, using mock data:', error);
        return mockChallenges.find(c => c.id === challengeId);
    }
}

// Make functions available globally
window.getChallengeById = getChallengeById;
window.renderChallenges = renderChallenges;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    renderChallenges();

    const sortSelect = document.getElementById('sortSelect');
    if (sortSelect) {
        sortSelect.addEventListener('change', () => {
            renderChallenges();
        });
    }

    document.getElementById('leaderboard-btn').onclick = function() {
        console.log('Opening leaderboard modal');
        document.getElementById('leaderboardModal').classList.add('active');
        document.body.style.overflow = 'hidden';
        updateLeaderboard(); 
    };

    document.querySelector('.leaderboard-modal-close').onclick = function() {
        document.getElementById('leaderboardModal').classList.remove('active');
        document.body.style.overflow = '';
    };

    document.querySelector('.leaderboard-modal-overlay').onclick = function() {
        document.getElementById('leaderboardModal').classList.remove('active');
        document.body.style.overflow = '';
    };
});