import os
import re


def fix_imports_in_file(file_path):
    """Replaces 'crackseg.src' with 'crackseg' in a given file."""
    try:
        with open(file_path, encoding="utf-8") as file:
            content = file.read()

        new_content, count = re.subn(
            r"from crackseg\.src", "from crackseg", content
        )

        if count > 0:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(new_content)
            print(f"Successfully fixed {count} import(s) in {file_path}")
            return True
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return False


def main():
    """Main function to fix imports in the gui directory."""
    gui_directory = "gui"
    fixed_files_count = 0
    for root, _, files in os.walk(gui_directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if fix_imports_in_file(file_path):
                    fixed_files_count += 1

    print(
        f"\nFinished processing. Fixed imports in {fixed_files_count} file(s)."
    )


if __name__ == "__main__":
    main()
