-- First, list all users (to see which ones you want to delete)
SELECT * FROM auth.users;

-- Delete specific test users
DELETE FROM auth.users 
WHERE email IN ('lazargabriel13@gmail.com', 'contact@gpudemystified.com');

-- Or to delete all users (be careful with this one!)
-- DELETE FROM auth.users;

-- To verify the deletion
SELECT * FROM auth.users;
