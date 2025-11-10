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
            
            let amountInDollars;
            if (selectedAmount === 'custom') {
                amountInDollars = parseFloat(document.getElementById('customAmount').value);
                if (!amountInDollars || amountInDollars < 1) {
                    alert('Please enter a valid amount (minimum $1)');
                    return;
                }
            } else {
                amountInDollars = parseFloat(selectedAmount);
            }
            
            // Convert to cents as integer
            const amountInCents = Math.round(amountInDollars * 100);
            
            // Disable button during processing
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            
            try {
                // Get user session
                const { data: { session } } = await window.supabaseClient.auth.getSession();
                
                // Prepare payload
                const payload = {
                    amount: amountInCents
                };
                
                // Only add email and name if they exist
                if (session?.user?.email) {
                    payload.email = session.user.email;
                }
                
                if (session?.user?.user_metadata?.full_name) {
                    payload.name = session.user.user_metadata.full_name;
                }
                
                console.log('Sending donation request:', payload);
                
                // Send to backend
                const response = await fetch(`${getApiUrl()}/create-donation-session`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(payload)
                });
                
                console.log('Response status:', response.status);
                
                if (!response.ok) {
                    const errorData = await response.json();
                    console.error('Error response:', errorData);
                    throw new Error(errorData.detail || 'Failed to create donation session');
                }
                
                const data = await response.json();
                console.log('Success response:', data);
                
                // Redirect to Stripe checkout
                if (data.checkout_url) {
                    window.location.href = data.checkout_url;
                } else {
                    throw new Error('No checkout URL returned');
                }
                
            } catch (error) {
                console.error('Error processing donation:', error);
                alert('Failed to process donation. Please try again.\n\n' + error.message);
                
                // Re-enable button
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="fas fa-heart"></i> Support Now';
            }
        });
    }
});

// Make it globally available
window.showDonationModal = showDonationModal;