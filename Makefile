export JULIA_CUDA_MEMORY_POOL:=none

main:
	julia ./src/main.jl ./params/config_server.toml

clean:
	rm -f *.txt *.png *.dat nohup.out
	rm -rf core

post:
	julia ./posts/post.jl ./params/config_server.toml

test:
	julia ./tests/runtests.jl ./params/config_server.toml

exact:
	julia ./tests/exact.jl

compare:
	julia ./tests/ed_from_kernel.jl

