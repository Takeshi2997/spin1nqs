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

const start_ref = Ref{UInt64}(time_ns())
function reset_start!()
    start_ref[] = time_ns()
end
function get_elapsed()
    return (time_ns() - start_ref[]) / 1e6
end
reset_start!()

function report(label)
    CUDA.synchronize()   # 非同期実行の完了を待つ (重要)
    time = get_elapsed()
    @printf("%-30s %.3f GiB, Elapsed time %5.3f ms\n", label, CUDA.used_memory() / 2^30, time)
    reset_start!()
end

include("ed_from_kernel.jl")
include("deterministic_sr.jl")
include("../src/hilbert.jl")
include("../src/sampler.jl")
include("../src/optimise.jl")

using .Hilbert
using .Model
using .Sampler
using .Physics
using .Optimise

function main()
    dirname = "./data/" * Dates.format(now(), "yyyymmdd")
    filename  = dirname * "/data.txt"
    mkpath(dirname)
    if isfile(filename)
        rm(filename)
    end
    touch(filename)
    commit = try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end
    open(filename, "a") do io
        @printf(io, "[%s] ==== 学習開始 （Git commit:%s）====\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), commit)
        @printf(io, "Epoch, UnixTime, Re<E>, Im<E>, VarE, Re<S2>, Im<S2>, <n1>, <n2>, <n3>, n_off, n_clipping,\n")
    end
 
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

    # 学習設定の読み込み
    train_config = config["training"]
    chunk = train_config["chunk"]
    n_walkers = train_config["n_walkers"]
    n_thermal = train_config["n_thermal"]
    n_steps = train_config["n_steps"]
    n_interval = train_config["n_interval"]
    n_epochs = train_config["n_epochs"]
    learning_rate = Float32(train_config["learning_rate"])
    epsilon = Float32(train_config["epsilon"])
    epsilon2 = Float32(train_config["epsilon2"])
    decay = Float32(train_config["decay"])
    lambda_min = Float32(train_config["lambda_min"])
    clipping_threshold = Float32(train_config["clipping_threshold"])
    n_total = n_walkers * n_steps

    # モデル設定の読み込み
    model_config = config["model"]
    hidden_dim = model_config["hidden_dim"]

    # IO設定の読み込み
    io_config = config["io"]
    log_iter = io_config["log_iter"]
    save_iter = io_config["save_iter"]

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

    println("=== VMCサンプリングを起動します ===")
    println("環境: ", CUDA.functional() ? "GPU (CUDA)" : "CPU (警告: 動作が遅くなります)")

    # === 2. 各種構造体・ネットワークの初期化 ===
    # A. ヒルベルト空間の確保と初期状態の配置 (M_z=0に固定)
    basis = MomentumSpinorBasis(k_max, n_particles, 256, n_walkers)
    initialize_states!(basis, target_Mz)

    # B. 複素数出力NQSモデルの構築 (出力2ch)
    nqs_model = build_momentum_nqs(k_max, hidden_dim=hidden_dim)
    ## ps_cpu, st_cpu = initialize_model(nqs_model, rng)
    ## e_start = 1
    ps_cpu, st_cpu = load_nqs_model("./data/20260725/nqs_model_4610_epoch7000.jld2")
    e_start = 7001
    n_params = Lux.parameterlength(ps_cpu)

    # 重み(ps)と状態(st)をGPUへ転送
    ps = ComponentArray(ps_cpu) |> cu
    st = st_cpu |> cu

    basis = enumerate_basis(N_PART)
    H_indep = build_H_independent(basis)

    run_deterministic(nqs_model, ps, st, basis, H_indep, N_MODES)
   
    println("=== 学習が正常に終了しました ===")
end

# 実行
main()
