install:
	poetry install --with dev,docs
	poetry run python -m spacy download en_core_web_sm

jupyter:
	poetry run jupyter lab

doc-check:
	poetry run pydocstyle biaslyze/

doc-preview: doc-check
	poetry run gendocs --config mkgendocs.yml
	poetry run mkdocs serve --dirtyreload

doc: doc-check
	poetry run gendocs --config mkgendocs.yml
	poetry run mkdocs build

style:
	poetry run isort .
	poetry run black .

typing:
	poetry run mypy biaslyze/

lint:
	poetry run ruff biaslyze/ --ignore E501

test:
	PYTHONPATH=biaslyze:$(PYTHONPATH) poetry run pytest --cov=biaslyze tests/

publish:
	poetry publish --build
