python setup.py check
python setup.py sdist

twine upload --repository-url https://test.pypi.org/legacy/ dist/*