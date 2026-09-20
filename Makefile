export JULIA_CUDA_MEMORY_POOL:=none

JULIA ?= julia --project
K := 5
C1 := 2.0

main:
	CUDA_VISIBLE_DEVICES=1 julia ./src/main.jl ./params/config_server.toml

clean:
	rm -f *.txt *.png *.dat nohup.out
	rm -rf core

data/ckpt_N8.jld2:
	$(JULIA) ./src/main_chain.jl --params ./params/config_server.toml --k_max $(K) --n 8 --c1 $(C1) --n_epoch 20000 \
	    --init fresh --out $@

data/ckpt_N12.jld2: data/ckpt_N8.jld2
	$(JULIA) ./src/main_chain.jl --params ./params/config_server.toml --k_max $(K) --n 12 --c1 $(C1) --n_epoch 10000 \
	    --init $< --out $@

data/ckpt_N16.jld2: data/ckpt_N12.jld2
	$(JULIA) ./src/main_chain.jl --params ./params/config_server.toml --k_max $(K) --n 16 --c1 $(C1) --n_epoch 10000 \
	    --init $< --out $@

data/ckpt_N20.jld2: data/ckpt_N16.jld2
	$(JULIA) ./src/main_chain.jl --params ./params/config_server.toml --k_max $(K) --n 20 --c1 $(C1) --n_epoch 10000 \
	    --init $< --out $@

data/ckpt_N24.jld2: data/ckpt_N20.jld2
	$(JULIA) ./src/main_chain.jl --params ./params/config_server.toml --k_max $(K) --n 24 --c1 $(C1) --n_epoch 10000 \
	    --init $< 

chain: data/ckpt_N24.jld2

post:
	julia ./posts/post.jl
	
postcompare:
	julia ./posts/post_compare.jl ./params/config_server.toml

test:
	julia ./tests/runtests.jl ./params/config_server.toml

exact:
	julia ./src/ed_main.jl

compare:
	julia ./tests/ed_from_kernel.jl

check:
	julia ./tests/check.jl ./params/config_server.toml

