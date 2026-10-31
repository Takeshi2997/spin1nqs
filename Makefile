export JULIA_CUDA_MEMORY_POOL:=none

JULIA ?= julia --project
K := 5
C1 := 0.2

main:
	CUDA_VISIBLE_DEVICES=0 julia ./src/main.jl ./params/config_server.toml

clean:
	rm -f *.txt *.png *.dat nohup.out
	rm -rf core
	
data/ckpt_N8.jld2:
	$(JULIA) ./src/main_chain.jl --params ./params/config_server.toml --k_max 2 --n 8 --c1 $(C1) --n_epochs 50000 \
	    --init fresh --out $@

data/ckpt_N12.jld2: data/ckpt_N8.jld2
	$(JULIA) ./src/main_chain.jl --params ./params/config_server.toml --k_max 3 --n 12 --c1 $(C1) --n_epochs 50000 \
	    --init data/ckpt_N8.jld2 --out $@

data/ckpt_N16.jld2: data/ckpt_N12.jld2
	$(JULIA) ./src/main_chain.jl --params ./params/config_server.toml --k_max 4 --n 16 --c1 $(C1) --n_epochs 50000 \
	    --init data/ckpt_N12.jld2  --out $@

data/ckpt_N20.jld2: data/ckpt_N16.jld2
	$(JULIA) ./src/main_chain.jl --params ./params/config_server.toml --k_max 5 --n 20 --c1 $(C1) --n_epochs 50000 \
	    --init data/ckpt_N16.jld2  --out $@

data/ckpt_N24.jld2: data/ckpt_N20.jld2
	$(JULIA) ./src/main_chain.jl --params ./params/config_server.toml --k_max 6 --n 24 --c1 $(C1) --n_epochs 50000 \
	    --init data/ckpt_N20.jld2 

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

