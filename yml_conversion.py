import yaml

def convert_requirements_to_yml(requirements_file, yml_file):
    with open(requirements_file, 'r') as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    conda_packages = []
    pip_packages = []

    # A simple (and incomplete) mapping of pip to conda names.
    # You might need to expand this.
    pip_to_conda_map = {
        "scikit-learn": "scikit-learn",
        "pandas": "pandas",
        "numpy": "numpy",
        # Add more mappings as needed
    }

    for package in packages:
        # Simple check if a conda equivalent exists in our map
        package_name = package.split('==')[0]
        if package_name in pip_to_conda_map:
            conda_packages.append(pip_to_conda_map[package_name])
        else:
            pip_packages.append(package)

    env_data = {
        'name': 'my-new-env',
        'channels': ['defaults', 'conda-forge'],
        'dependencies': conda_packages
    }

    if pip_packages:
        env_data['dependencies'].append('pip')
        env_data['dependencies'].append({'pip': pip_packages})

    with open(yml_file, 'w') as f:
        yaml.dump(env_data, f, default_flow_style=False)

# Usage
convert_requirements_to_yml('requirements.txt', 'environment.yml')