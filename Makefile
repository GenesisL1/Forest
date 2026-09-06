.PHONY: cpp test test-mint test-deployment test-server test-wheel test-formal test-archive test-science test-ui test-contracts test-evm benchmark benchmark-evm evidence-parity evidence-evm freeze-evidence pdfs verify

SOURCE_DATE_EPOCH ?= 1788652800
TZ := UTC
export SOURCE_DATE_EPOCH
export FORCE_SOURCE_DATE = 1
export TZ

cpp:
	./build_cpp_trainer.sh

test:
	python3 -m unittest -v tests.publication.test_publication

test-mint:
	python3 -m unittest -v tests.publication.test_mint_validation tests.publication.test_mint_workflow

test-deployment:
	python3 -m unittest -v tests.publication.test_deployment_identity

test-server:
	python3 -m unittest -v tests.test_local_trainer_server

test-wheel:
	python3 -m unittest -v tests.publication.test_wheel_package

test-formal:
	python3 -m unittest -v tests.contracts_publication.test_formal_properties

test-archive:
	python3 -m unittest -v tests.publication.test_independent_archive

test-science:
	python3 tests/scientific_invariants.py --public

test-ui:
	python3 tests/ui_static_check.py
	node tests/ui_seed_parameters.mjs

test-contracts:
	node tests/contracts_publication/compile_contracts.mjs

test-evm:
	node tests/contracts_publication/test_evm_integration.mjs

benchmark:
	python3 benchmarks/publication_benchmark.py --out benchmarks/results/publication_benchmark.json

benchmark-evm:
	node benchmarks/evm_scaling_benchmark.mjs

evidence-parity:
	python3 benchmarks/generate_parity_evidence.py --out benchmarks/results/parity_matrix.json

evidence-evm:
	node tests/contracts_publication/test_evm_integration.mjs --out benchmarks/results/evm_integration.json

freeze-evidence:
	$(MAKE) evidence-parity
	$(MAKE) evidence-evm

pdfs:
	./paper/build_pdfs.sh

verify: test-mint test-deployment test-server test-wheel test test-formal test-archive test-science test-ui test-contracts test-evm
