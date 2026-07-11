.PHONY: test

main:
	julia ./src/main.jl ./params/config_server.toml
clean:
	rm -f *.txt *.png *.dat nohup.out
	rm -rf core
test:
	julia ./test/runtests.jl ./params/config_server.toml

exact:
	julia ./test/exact.jl

compare:
	julia ./test/ed_from_kernel.jl

