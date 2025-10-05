const mockChallenges = [
    {
        id: "vector_addition",
        title: "Vector Addition",
        description: "Add two vectors element-wise on the GPU.",
        tags: ["CUDA", "Beginner"],
        code: "#include <cuda_runtime.h> \n\n__global__ void vectorAdd(float *A, float *B, float *C, int N) {\n    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < N) C[i] = A[i] + B[i];\n}"
    },
    {
        id: "matrix_multiplication",
        title: "Matrix Multiplication",
        description: "Multiply two matrices using shared memory optimization.",
        tags: ["CUDA", "Intermediate"],
        code: "__global__ void matMul(float *A, float *B, float *C, int N) {\n    int row = blockIdx.y * blockDim.y + threadIdx.y;\n    int col = blockIdx.x * blockDim.x + threadIdx.x;\n    if (row < N && col < N) {\n        float sum = 0;\n        for (int k = 0; k < N; ++k)\n            sum += A[row * N + k] * B[k * N + col];\n        C[row * N + col] = sum;\n    }\n}"
    },
    {
        id: "image_convolution",
        title: "Image Convolution",
        description: "Implement a 2D image convolution kernel.",
        tags: ["CUDA", "Advanced", "Image Processing"],
        code: "__global__ void conv2d(float *input, float *kernel, float *output, int width, int height, int ksize) {\n    int x = blockIdx.x * blockDim.x + threadIdx.x;\n    int y = blockIdx.y * blockDim.y + threadIdx.y;\n    if (x >= width || y >= height) return;\n    float sum = 0;\n    for (int ky = 0; ky < ksize; ky++) {\n        for (int kx = 0; kx < ksize; kx++) {\n            int ix = x + kx - ksize / 2;\n            int iy = y + ky - ksize / 2;\n            if (ix >= 0 && iy >= 0 && ix < width && iy < height)\n                sum += input[iy * width + ix] * kernel[ky * ksize + kx];\n        }\n    }\n    output[y * width + x] = sum;\n}"
    }
];

function renderChallenges() {
    const grid = document.getElementById('challenges-grid');
    grid.innerHTML = mockChallenges.map(challenge => `
        <div class="challenge-card" data-id="${challenge.id}">
            <h3>${challenge.title}</h3>
            <p>${challenge.description}</p>
            <div class="challenge-tags">
                ${challenge.tags.map(tag => `
                    <span class="tag ${tag.toLowerCase()}">${tag}</span>
                `).join('')}
            </div>
        </div>
    `).join('');

    // Add click handlers
    document.querySelectorAll('.challenge-card').forEach(card => {
        card.addEventListener('click', () => openChallenge(card.dataset.id));
    });
}

document.addEventListener('DOMContentLoaded', renderChallenges);