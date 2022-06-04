codestyle:
	pip install -U isort autopep8 flake8
	autopep8 --in-place -r .
	flake8 --config=setup.cfg