-- Drop existing triggers, functions, and tables
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
DROP FUNCTION IF EXISTS handle_new_user() CASCADE;
DROP TABLE IF EXISTS challenges CASCADE;
DROP TABLE IF EXISTS profiles CASCADE;

-- Create a simple profiles table
CREATE TABLE profiles (
    id uuid references auth.users on delete cascade primary key,
    email text,
    is_pro boolean DEFAULT false,
    hints_count integer DEFAULT 10,
    submissions_count integer DEFAULT 50
);

-- Create a challenges table
CREATE TABLE challenges (
    id text primary key,
    title text not null,
    description text not null,
    short_description text,
    points integer not null,
    tags jsonb,
    flags jsonb,
    profile jsonb,
    timeout integer,
    initial_code text,
    template_code text,
    expected_results jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Enable Row Level Security (RLS)
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE challenges ENABLE ROW LEVEL SECURITY;

-- Create policies for profiles
CREATE POLICY "Users can view their own profile"
    ON profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
    ON profiles FOR UPDATE
    USING (auth.uid() = id);

-- Create policy for challenges (all authenticated users can view challenges)
CREATE POLICY "Authenticated users can view challenges"
    ON challenges FOR SELECT
    USING (auth.role() = 'authenticated');

-- Create the function to handle new user creation
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS trigger
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO public.profiles (id, email, is_pro, hints_count)
    VALUES (new.id, new.email, false, 50)
    ON CONFLICT (id) DO NOTHING;
    RETURN new;
END;
$$ LANGUAGE plpgsql;

-- Create the trigger to automatically create profiles
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user();

-- Insert sample challenges
INSERT INTO challenges (id, title, description, short_description, points, tags, flags, profile, timeout, initial_code, template_code, expected_results) VALUES
(
    'challenge_1',
    'Matrix Multiplication Kernel',
    'Write a CUDA kernel for matrix multiplication',
    'Implement basic matrix multiplication using CUDA kernels',
    100,
    '["beginner", "cuda"]',
    '["-arch=sm_75", "-O3"]',
    '{"enabled": false}',
    5,
    '__global__ void matrixMul(float* A, float* B, float* C, int width) {
    // TODO: Write your matrix multiplication kernel here
    // Hint: Use 2D thread indexing with blockIdx and threadIdx
    // C[row][col] = sum(A[row][k] * B[k][col]) for k = 0 to width-1
}',
    '-- Template code here --',  -- You'll want to store the full template code
    '{
        "kernels": [{
            "expected_duration_ms": 2.0,
            "tolerance_percent": 100,
            "expected_grid_size": {"x": 32, "y": 32, "z": 1},
            "expected_block_size": {"x": 16, "y": 16, "z": 1}
        }],
        "memory_ops": [
            {"type": "HtoD", "expected_count": 2, "expected_size_mb": 1.0},
            {"type": "DtoH", "expected_count": 1, "expected_size_mb": 1.0}
        ]
    }'
);