import os
import json
import codecs

def find_metadata_files(repo_paths):
    metadata_files = []

    # Recursively search for metadata.json files in the specified repository paths
    for repo_path in repo_paths:
        for root, dirs, files in os.walk(repo_path):
            if 'metadata.json' in files:
                metadata_files.append(os.path.join(root, 'metadata.json'))

    return metadata_files

def extract_readme(file_path):
    # Check for the README.md file in the current directory
    readme_path = os.path.join(os.path.dirname(file_path), 'README.md')
    if os.path.exists(readme_path):
        with codecs.open(readme_path, 'r', 'utf-8') as readme_file:
            return readme_file.read()
    else:
        # If README.md is not found, look for it one level up
        readme_path = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'README.md')
        if os.path.exists(readme_path):
            with codecs.open(readme_path, 'r', 'utf-8') as readme_file:
                return readme_file.read()
        else:
            return ''

def extract_application_name(readme_path):
    # Extract the application name from the README file path
    parts = readme_path.split(os.sep)
    if 'applications' in parts:
        index = parts.index('applications')
        if index + 1 < len(parts):
            return parts[index + 1]
    elif 'operators' in parts:
        index = parts.index('operators')
        if index + 1 < len(parts):
            return parts[index + 1]
    return ''

def gather_metadata(repo_path, output_file):
    metadata_files = find_metadata_files(repo_path)
    metadata = []

    # Iterate over the found metadata files
    for file_path in metadata_files:
        with open(file_path, 'r') as file:
            data = json.load(file)

            if "application" in data:
                data["metadata"] = data.pop("application")
            elif "operator" in data:
                data["metadata"] = data.pop("operator")
            
            readme = extract_readme(file_path)
            application_name = extract_application_name(file_path)
            source_folder = 'applications' if 'applications' in file_path else 'operators'
            data['readme'] = readme
            data['application_name'] = application_name
            data['source_folder'] = source_folder
            metadata.append(data)

    # Write the metadata to the output file
    with open(output_file, 'w') as output:
        json.dump(metadata, output, indent=4)

# Specify the repository path and the output file name
repo_paths = ['holohub/applications', 'holohub/operators']
output_file = 'aggregate_metadata.json'

# Call the function to gather metadata
gather_metadata(repo_paths, output_file)