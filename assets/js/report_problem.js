// Show report problem modal
function showReportModal() {
    const modal = document.getElementById('reportProblemModal');
    modal.style.display = 'flex';
    
    // Auto-fill current page
    const pageInput = document.getElementById('reportPage');
   
    // Auto-fill email if user is logged in
    const emailInput = document.getElementById('reportEmail');
    if (window.userProfile?.email) {
        emailInput.value = window.userProfile.email;
    }
}

// Handle report submission
document.addEventListener('DOMContentLoaded', () => {
    const reportBtn = document.getElementById('reportProblemBtn');
    const reportForm = document.getElementById('reportProblemForm');
    
    if (reportBtn) {
        reportBtn.onclick = showReportModal;
    }
    
    if (reportForm) {
        reportForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = e.target.querySelector('.auth-submit-btn');
            const messageDiv = document.getElementById('reportMessage');
            
            // Clear previous messages
            messageDiv.className = 'report-message';
            messageDiv.textContent = '';
            
            // Disable button during submission
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Submitting...';
            
            try {
                const reportData = {
                    type: document.getElementById('reportType').value,
                    page: document.getElementById('reportPage').value,
                    description: document.getElementById('reportDescription').value,
                    email: document.getElementById('reportEmail').value || 'anonymous',
                    url: window.location.href,
                    timestamp: new Date().toISOString()
                };
                
                // Send to your backend
                const response = await fetch(`${getApiUrl()}/report-problem`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(reportData)
                });
                
                if (!response.ok) {
                    throw new Error('Failed to submit report');
                }
                
                // Success
                messageDiv.className = 'report-message success';
                messageDiv.textContent = '✓ Thank you! Your report has been submitted successfully.';
                
                // Reset form after 2 seconds
                setTimeout(() => {
                    reportForm.reset();
                    setTimeout(() => {
                        document.getElementById('reportProblemModal').style.display = 'none';
                        messageDiv.className = 'report-message';
                        messageDiv.textContent = '';
                    }, 1000);
                }, 2000);
                
            } catch (error) {
                console.error('Error submitting report:', error);
                messageDiv.className = 'report-message error';
                messageDiv.textContent = '✗ Failed to submit report. Please try again or email us directly.';
            } finally {
                // Re-enable button
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="fas fa-paper-plane"></i> Submit Report';
            }
        });
    }
});

// Make it globally available
window.showReportModal = showReportModal;