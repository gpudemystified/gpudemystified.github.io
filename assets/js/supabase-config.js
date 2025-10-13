const supabaseUrl = 'https://fiugbhezwngkzxypdtvw.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZpdWdiaGV6d25na3p4eXBkdHZ3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAyMDY4NjAsImV4cCI6MjA3NTc4Mjg2MH0.p-_Fr1RJRxelBPx9_-sroEhYuD1L1fAnD-2Isb8XdAA';

const supabaseClient = supabase.createClient(supabaseUrl, supabaseKey)

// Export for use in other files
window.supabaseClient = supabaseClient;