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

# 自作モジュールの読み込み（同じディレクトリにあると仮定）
include("hilbert.jl")
include("model.jl")
include("sampler.jl")
include("physics.jl")

using .Hilbert
using .Model
using .Sampler
using .Physics

function main()
    dirname = "./data/" * Dates.format(now(), "yyyymmdd") * "_estimate"
    mkpath(dirname)
    if isfile(filename)
        rm(filename)
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
    estimate_config = config["estimate"]
    n_walkers = estimate_config["n_walkers"]
    n_thermal = estimate_config["n_thermal"]

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

    println("=== VMCサンプリングを起動します ===")
    println("環境: ", CUDA.functional() ? "GPU (CUDA)" : "CPU (警告: 動作が遅くなります)")

    # === 2. 各種構造体・ネットワークの初期化 ===
    # A. ヒルベルト空間の確保と初期状態の配置 (M_z=0に固定)
    basis = MomentumSpinorBasis(k_max, n_particles, 256, n_walkers)
    initialize_states!(basis, target_Mz)

    # B. 複素数出力NQSモデルの構築 (出力2ch)
    nqs_model = build_momentum_nqs(k_max, hidden_dim=hidden_dim)
    ps_cpu, st_cpu = load_nqs_model("./data/20260709/nqs_model_770_epoch20000.jld2")
    
    # 重み(ps)と状態(st)をGPUへ転送
    ps = ComponentArray(ps_cpu) |> cu
    st = st_cpu |> cu

    # C. サンプラーバッファの確保
    sampler = MCMCSampler(basis)
    buffer = PhysicsBuffer(k_max, min(n_particles, 3 * (2 * k_max + 1))^2 * 3 * (2 * k_max + 1), n_walkers)

    # === 3. マルコフ連鎖の熱平衡化（Thermalization） ===
    println("マルコフ連鎖を熱平衡化中 ($(n_thermal) ステップ)...")
    for step in 1:n_thermal
        sample_step!(sampler, basis, nqs_model, k_max, n_particles, ps, st)
    end
    println("熱平衡化が完了しました。")

    # === 4. 推定 ===
    # サンプリングとデータ収集
    # 現在の状態でのネットワーク出力を取得 [2, n_walkers]
    inputs = basis.states
    outputs = eval_complex_network(nqs_model, inputs, ps, st)

    # 局所エネルギー E_loc の計算
    E_loc = compute_local_energy(basis.states, outputs, buffer.proposed_states, buffer.matrix_elements, params, basis.threads, nqs_model, ps, st)

    # エネルギー期待値・粒子数期待値の算出
    E_mean = sum(E_loc) / n_walkers
    E_real = real(E_mean)
    E_imag = imag(E_mean)
    n1_mean = sum(basis.states[:, 1, :]) / n_walkers
    n2_mean = sum(basis.states[:, 2, :]) / n_walkers
    n3_mean = sum(basis.states[:, 3, :]) / n_walkers
    @printf("E_real, E_imag, n1_mean, n2_mean, n3_mean, \n")
    @printf("%10.5f, %10.5f, %6.3f, %6.3f, %6.3f,\n", E_real, E_imag, n1_mean, n2_mean, n3_mean)

    # 相関関数の評価
    eval_space_correlation(basis.states, outputs, k_max, basis.threads, n_walkers, nqs_model, ps, st, dirname)

    println("=== 計算が終了しました ===")
end

function eval_space_correlation(states, outputs, k_max, threads, n_walkers, nqs_model, ps, st, dirname)
    filename  = dirname * "/space_correlation.txt"
    if isfile(filename)
        rm(filename)
    end
    touch(filename)

    # 現在の 186-195 行目を以下で置き換え
    # rho2,s(q=0) = <N_s (N_s - 1)>, N_s = sum_k n_{k,s}
    # ψ を変えない全項の和なので、占有数だけから直接計算できる。
    Ns_w = dropdims(sum(states, dims=1), dims=1)
    Ns_diag = Array(dropdims(sum(Ns_w .* (Ns_w .- Int32(1)), dims=2), dims=2) ./ n_walkers)
    n1_diag = Float32(Ns_diag[1])
    n2_diag = Float32(Ns_diag[2])
    n3_diag = Float32(Ns_diag[3])
    @printf("n1_diag, n2_diag, n3_diag, \n") 
    @printf("%6.3f, %6.3f, %6.3f,\n", n1_diag, n2_diag, n3_diag)
   
    # 新しい compute_local_correlation の戻り値は [n_modes(=q平面), 3, n_walkers]。
    # ウォーカー方向 (dims=3) で平均する (旧レイアウト [n_walkers, n_modes, 3] では dims=1 だった)
    rho2_q_loc = compute_local_correlation(states, outputs, k_max, threads, nqs_model, ps, st)
    rho2_q_mean = Array(dropdims(sum(rho2_q_loc, dims=3), dims=3) ./ n_walkers)   # [n_modes, 3]
    rho2_q_1 = rho2_q_mean[:, 1]
    rho2_q_2 = rho2_q_mean[:, 2]
    rho2_q_3 = rho2_q_mean[:, 3]
    rho2_q_1[k_max + 1] = n1_diag
    rho2_q_2[k_max + 1] = n2_diag
    rho2_q_3[k_max + 1] = n3_diag

    # フーリエ変換
    L_box = Float32(2 * π)
    x_grid = Float32.(range(-L_box/2, L_box/2, length=1000))
    k_list = Float32.((2 * π / L_box) .* (-k_max:k_max))
    W = exp.(-1.0f0im .* x_grid .* k_list')
    cor1_x_vec = real.(W * rho2_q_1) ./ L_box
    cor2_x_vec = real.(W * rho2_q_2) ./ L_box
    cor3_x_vec = real.(W * rho2_q_3) ./ L_box

    open(filename, "a") do io
        @printf(io, "x, <n1>, <n2>, <n3>,\n")
    end
    for x in 1:1000
        cor1_x = cor1_x_vec[x]
        cor2_x = cor2_x_vec[x]
        cor3_x = cor3_x_vec[x]
        open(filename, "a") do io
            @printf(io, "%6.3f, %6.9f, %6.9f, %6.9f,\n", x_grid[x], cor1_x, cor2_x, cor3_x)
        end
    end

    return nothing
end

# 実行
main()