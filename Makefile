clean:
	rm -rf build/

%.ipynb: %.py
	mkdir -p build/
	uv run jupytext $^ -o $@

build/%.html: %.ipynb
	uv run jupyter nbconvert --output-dir=build $^ --to html --execute


