from ingestion.clone_repo import clone_repository
from ingestion.scan_files import scan_repository
from parsing.extract_functions import extract_functions
from embeddings.embed_functions import generate_embeddings
from bug_detector.detect_patterns import detect_issues
from patch_generator.generate_patch import generate_patches

def main():
    repo_url = "https://github.com/pallets/flask"  # example

    print("Cloning repo...")
    repo_path = clone_repository(repo_url)

    print("Scanning files...")
    files = scan_repository(repo_path)

    print("Extracting functions...")
    functions = extract_functions(files)

    print("Generating embeddings...")
    embeddings = generate_embeddings(functions)

    print("Detecting issues...")
    issues = detect_issues(functions, embeddings)

    print("Generating patches...")
    patches = generate_patches(issues)

    print("Saving results...")
    # save to analysis_results/

    print("Done!")

if __name__ == "__main__":
    main()