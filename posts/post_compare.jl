using TOML
using LinearAlgebra
using CUDA
using Random
using Lux
using Optimisers
using Zygote
using ComponentArrays
using Printf
using Dates
using Statistics
using StatsBase

# 自作モジュールの読み込み
include("../src/hilbert.jl")
include("../tests/ed_from_kernel.jl")
include("../src/sampler.jl")
include("../src/physics.jl")
include("../src/model.jl")

using .Hilbert
using .Model
using .Sampler
using .Physics

function main()
    # === 1. 物理・シミュレーションパラメータの設定 ===
    config_path = length(ARGS) > 0 ? ARGS[1] : "config_local.toml"
    println("🔧 Loading configuration from: ", config_path)
    
    # 2. TOMLファイルのパース
    config = TOML.parsefile(config_path)

    # システム設定の読み込み
    sys_config = config["system"]
    k_max = sys_config["k_max"]
    n_particles = sys_config["n_particles"]
    hbar2_over_2m = Float32(sys_config["hbar2_over_2m"])
    c0 = Float32(sys_config["c0"])
    c1 = Float32(sys_config["c1"])
    target_Mz = sys_config["target_Mz"]

    # 推定設定の読み込み
    estimate_config = config["estimate"]
    n_walkers = estimate_config["n_walkers"]
    n_steps = estimate_config["n_steps"]
    beta = estimate_config["beta"]
    n_total = n_walkers * n_steps

    # モデル設定の読み込み
    model_config = config["model"]
    hidden_dim = model_config["hidden_dim"]

    # ハミルトニアン係数（接触相互作用）
    params = SystemParams(
        k_max,
        2 * k_max + 1,
        hbar2_over_2m,
        c0,  # c0 (密度相互作用)
        c1   # c1 (スピン交換相互作用)
    )

    rng = Xoshiro(42)
    CUDA.allowscalar(false) # GPUのシリアルアクセス(低速化の原因)を禁止してデバッグ

    println("=== NQSポスト処理を起動します ===")
    println("環境: ", CUDA.functional() ? "GPU (CUDA)" : "CPU (警告: 動作が遅くなります)")

    # === 2. 各種構造体・ネットワークの初期化 ===
    # A. ヒルベルト空間の確保と初期状態の配置 (M_z=0に固定)
    basis = MomentumSpinorBasis(k_max, n_particles, 256, n_walkers)
    initialize_states!(basis, target_Mz)

    # B. 複素数出力NQSモデルの構築 (出力2ch)
    nqs_model = build_momentum_nqs(k_max, hidden_dim=hidden_dim)
    srcdir = "./data/20260809/"
    epoch = 50000
    filename = "nqs_model_4610_epoch" * string(epoch) * ".jld2"
    ps_cpu, st_cpu = load_nqs_model(srcdir * filename)

    ## == 厳密対角化との比較 ==
    basis = enumerate_basis(N_PART)
    inputs_flat = hcat([occ for occ in basis]...)
    inputs = reshape(inputs_flat, N_MODES, 3, :)  # [11, 3, 23607]
    log_psi = eval_complex_network(nqs_model, inputs, ps_cpu, st_cpu)
    max_log_psi = maximum(real.(log_psi))
    psi = exp.(log_psi .- max_log_psi)
    psi ./= sqrt(sum(abs2, psi))
    H_indep = build_H_independent(basis)
    E_exact = real(dot(psi, H_indep * psi))

    dim = length(basis)

    # <N_s>
    ns_mean = zeros(3)
    for (i, occ) in enumerate(basis)
        w = abs2(psi[i])
        for s in 1:3
            ns = sum(Int(occ[cell_index(m, s)]) for m in 1:N_MODES)
            ns_mean[s] += w * ns
        end
    end

    # <N_off>
    n_off = 0
    for (i, occ) in enumerate(basis)
        w = abs2(psi[i])
        for s in 1:3
            n_off += w * occ[cell_index(K_MAX + 1, s)]
        end
    end
    n_off = N_PART - n_off

    @printf("E_real, n1_mean, n2_mean, n3_mean, n_off,\n")
    @printf("%6.8f, %6.8f, %6.8f, %6.8f, %6.8f,\n", E_exact, ns_mean[1], ns_mean[2], ns_mean[3], n_off)
 
    vals, vecs, _ = eigsolve(H_indep, dim, 1, :SR; issymmetric = true, tol = 1e-12)
    err = psi .- vecs[1] * sign(dot(vecs[1], psi))              # 位相を合わせて差分
    idx = sortperm(abs.(err), rev=true)[1:20]
    for i in idx
        occ = reshape(basis[i], N_MODES, 3)
        @printf("|c_ED|=%.5f |c_NQS|=%.5f 位相差=%.3f rad  E_kin=%d\n",
                abs(vecs[1][i]), abs(psi[i]), angle(psi[i] / vecs[1][i]),
                sum(occ[m,s]*(m-k_max-1)^2 for m in 1:N_MODES, s in 1:3))
    end
end

# 実行
main()