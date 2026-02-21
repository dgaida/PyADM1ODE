import os
import re
import sys
from datetime import date

def bump_version(current_version):
    parts = current_version.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {current_version}")
    major, minor, patch = map(int, parts)
    patch += 1
    return f"{major}.{minor}.{patch}"

def update_file(filepath, pattern, replacement, flags=0):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content, count = re.subn(pattern, replacement, content, flags=flags)
    if count == 0:
        print(f"Warning: No matches found for pattern in {filepath}")
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath} ({count} replacements)")

def main():
    pyproject_path = 'pyproject.toml'
    if not os.path.exists(pyproject_path):
        print(f"Error: {pyproject_path} not found.")
        sys.exit(1)

    with open(pyproject_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Specifically look for version in [project] section
    match = re.search(r'\[project\]\n(?:.*\n)*?version\s*=\s*"([^"]*)"', content)
    if not match:
        # Fallback to general search if section not found exactly as expected
        match = re.search(r'^version\s*=\s*"([^"]*)"', content, re.MULTILINE)

    if not match:
        print("Error: Could not find version in pyproject.toml")
        sys.exit(1)

    current_version = match.group(1)
    new_version = bump_version(current_version)
    today_iso = date.today().isoformat()
    today_german = date.today().strftime('%d.%m.%Y')

    print(f"Bumping version: {current_version} -> {new_version}")

    # pyproject.toml: only update version in [project] section
    update_file('pyproject.toml',
                r'(\[project\]\n(?:.*\n)*?version\s*=\s*)"[^"]*"',
                r'\g<1>"' + new_version + '"')

    # Other files
    updates = [
        ('pyadm1/__version__.py', r'(__version__\s*=\s*)"[^"]*"', r'\g<1>"' + new_version + '"'),
        ('tests/__init__.py', r'(__version__\s*=\s*)"[^"]*"', r'\g<1>"' + new_version + '"'),
        ('docs/en/metrics.md', r'(\*\*Changelog\*\*:\s*Up to date \(v)[^)]*(\))', r'\g<1>' + new_version + r'\g<2>'),
        ('docs/en/metrics.md', r'(\*\*Last Updated\*\*:\s*)\d{4}-\d{2}-\d{2}', r'\g<1>' + today_iso),
        ('docs/de/metrics.md', r'(\*\*Changelog\*\*:\s*Aktuell \(v)[^)]*(\))', r'\g<1>' + new_version + r'\g<2>'),
        ('docs/de/metrics.md', r'(\*\*Zuletzt aktualisiert\*\*:\s*)\d{2}\.\d{2}\.\d{4}', r'\g<1>' + today_german),
    ]

    for filepath, pattern, replacement in updates:
        update_file(filepath, pattern, replacement)

    github_output = os.environ.get('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f"new_version={new_version}\n")

    print(f"Successfully bumped version to {new_version}")

if __name__ == "__main__":
    main()
