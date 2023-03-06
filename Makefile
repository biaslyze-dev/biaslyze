install:
	poetry install

jupyter:
	poetry run jupyter lab

doc-preview:
	poetry run gendocs --config mkgendocs.yml
	poetry run mkdocs serve --dirtyreload

doc:
	poetry run gendocs --config mkgendocs.yml
	poetry run mkdocs build

style:
	poetry run isort .
	poetry run black .

lint:
	poetry run ruff biaslyze/ --ignore E501

test:
	PYTHONPATH=biaslyze:$(PYTHONPATH) poetry run pytest --cov=biaslyze tests/

