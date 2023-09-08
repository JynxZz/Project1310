import os
import subprocess

# Define the directory where you want to start the operation
base_dir = "/Users/jynxzz_air/code/lewagon-projects/project-1310"

# Function to count files and lines of code
def count_files_and_lines(directory):
    total_files = 0
    total_lines = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):  # You can specify other file extensions if needed
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                total_files += 1
                total_lines += len(lines)

    return total_files, total_lines

# Change to the base directory
os.chdir(base_dir)
total_lines = 0
total_files = 0
# Iterate through all subdirectories
for root, dirs, _ in os.walk("."):
    if ".git" in dirs:
        repo_name = os.path.basename(root)

        # Pull the latest changes from the Git repository
        print(f"Pulling {repo_name} repo")
        subprocess.run(["git", "pull"], cwd=root)

        # Count the number of files and lines of code
        nb_files, nb_lines = count_files_and_lines(root)
        total_lines += nb_lines
        total_files += nb_files
        # Print the results
        print(f"Files: {nb_files}")
        print(f"Coding lines: {nb_lines}")
        print()


# Print the final results
print(f"Files: {total_files}")
print(f"Coding lines: {total_lines}")
