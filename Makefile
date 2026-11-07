.PHONY: install test lint format demo verify strategy real docker clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

demo:
	f1lap run --synthetic --out /tmp/f1lap-demo

verify:
	f1lap verify --n-races 60 --out docs/img

strategy:
	f1lap strategy --year 2024 --race Bahrain --pit-loss 22 --out results

real:
	f1lap run --year 2024 --races Bahrain Spain Hungary --out results

docker:
	docker build -t f1lap . && docker run --rm -v $(PWD)/results:/results f1lap run --synthetic

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
