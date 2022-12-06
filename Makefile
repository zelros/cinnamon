build:
	python3 -m build

test:
	python3 -m pytest src/cinnamon

pypi-upload:
	python3 -m twine upload --repository pypi dist/*