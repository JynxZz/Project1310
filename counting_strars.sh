#!/bin/bash

# Define the directory where you want to start the operation
base_dir="/Users/jynxzz_air/code/lewagon-projects/project-1310"

# Change to the base directory
cd "$base_dir" || exit

# Iterate through all subdirectories
for dir in $(find . -type d -name ".git" -exec dirname {} \;); do
  # Get the repository name by removing the leading "./"
  repo_name="${dir#./}"

  # Pull the latest changes from the Git repository
  echo "Pulling $repo_name repo"
  (cd "$dir" && git pull)

  # Count the number of files and lines of code
  nb_files=$(find "$dir" -type f | wc -l)
  nb_lines=$(find "$dir" -type f -exec cat {} \; | wc -l)

  # Print the results
  echo "Files: $nb_files"
  echo "Coding lines: $nb_lines"
  echo
done
