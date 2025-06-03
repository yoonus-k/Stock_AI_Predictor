
from pip._internal.operations import freeze

# Get installed packages without versions
packages_without_versions = [pkg.split('==')[0] for pkg in freeze.freeze()]

# Write to requirements.txt
with open('requirements.txt', 'w') as f:
    f.write('\n'.join(packages_without_versions))