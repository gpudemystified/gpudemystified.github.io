// Show donation modal
function showDonationModal() {
    const modal = document.getElementById('donationModal');
    modal.style.display = 'flex';
}

// Handle donation form
document.addEventListener('DOMContentLoaded', () => {
    const donationBtn = document.getElementById('donationBtn');
    const donationForm = document.getElementById('donationForm');
    const customAmountInput = document.getElementById('customAmountInput');
    
    if (donationBtn) {
        donationBtn.onclick = showDonationModal;
    }
    
    // Show/hide custom amount input
    document.querySelectorAll('input[name="amount"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'custom') {
                customAmountInput.style.display = 'block';
                document.getElementById('customAmount').focus();
            } else {
                customAmountInput.style.display = 'none';
            }
        });
    });
    
    if (donationForm) {
        donationForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = e.target.querySelector('.donation-submit-btn');
            const selectedAmount = document.querySelector('input[name="amount"]:checked').value;
            
            let amount;
            if (selectedAmount === 'custom') {
                amount = parseFloat(document.getElementById('customAmount').value);
                if (!amount || amount < 1) {
                    alert('Please enter a valid amount (minimum $1)');
                    return;
                }
            } else {
                amount = parseFloat(selectedAmount);
            }
            
            // Disable button during processing
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            
            try {
                // Get user session
                const { data: { session } } = await window.supabaseClient.auth.getSession();
                
                // Send to backend
                const response = await fetch(`${getApiUrl()}/create-donation-session`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        amount: amount,
                        user_id: session?.user?.id || null,
                        email: session?.user?.email || null
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to create donation session');
                }
                
                const data = await response.json();
                
                // Redirect to Stripe checkout
                if (data.checkout_url) {
                    window.location.href = data.checkout_url;
                } else {
                    throw new Error('No checkout URL returned');
                }
                
            } catch (error) {
                console.error('Error processing donation:', error);
                alert('Failed to process donation. Please try again.');
                
                // Re-enable button
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="fas fa-heart"></i> Support Now';
            }
        });
    }
});

// Make it globally available
window.showDonationModal = showDonationModal;