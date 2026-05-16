import os
import re
import sys

def fix_markdown_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified = False
    new_lines = []

    # Pattern for markdown lists: starts with -, *, +, or 1., 2. etc.
    list_pattern = re.compile(r'^(\s*)([-*+]|\d+\.)\s+')

    for line in lines:
        stripped_line = line.rstrip('\n')

        # Check if it's a list item and doesn't already end with two spaces
        if list_pattern.match(stripped_line):
            if not stripped_line.endswith('  '):
                stripped_line = stripped_line.rstrip() + '  '
                modified = True

        new_lines.append(stripped_line + '\n')

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Fixed lists in: {filepath}")

def main(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                fix_markdown_file(os.path.join(root, file))

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else 'docs'
    main(target_dir)
