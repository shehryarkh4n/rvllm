PY ?= python3

.PHONY: install port emit status verify clean

install:
	$(PY) -m pip install -e .

port emit:
	$(PY) -m harness.emit_all

status:
	$(PY) -m harness.emit_all --status

verify:
	$(PY) -m harness.verify_all

clean:
	rm -rf out/*.mlir
