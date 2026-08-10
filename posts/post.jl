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
    dirname = "./data/" * Dates.format(now(), "yyyymmdd") * "_estimated/"
    if !isdir(dirname)
        mkpath(dirname)
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

    # 推定設定の読み込み
    estimate_config = config["estimate"]
    n_walkers = estimate_config["n_walkers"]
    n_steps = estimate_config["n_steps"]
    n_thermal = estimate_config["n_thermal"]
    n_interval = estimate_config["n_interval"]
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
    cp(srcdir * filename, dirname * filename, force=true)
    ps_cpu, st_cpu = load_nqs_model(dirname * filename)

    filename  = dirname * "/data_log_epoch" * string(epoch) * ".txt"
    io = open(filename, "w")
   

    # 重み(ps)と状態(st)をGPUへ転送
    ps = ComponentArray(ps_cpu) |> cu
    st = st_cpu |> cu

    # C. サンプラーバッファの確保
    sampler = MCMCSampler(basis)
    buffer = PhysicsBuffer(k_max, min(n_particles, 3 * (2 * k_max + 1))^2 * 3 * (2 * k_max + 1), n_walkers)
 
    # === 3. マルコフ連鎖の熱平衡化（Thermalization） ===
    println("マルコフ連鎖を熱平衡化中 ($(n_thermal) ステップ)...")
    for step in 1:n_thermal
        sample_step!(sampler, basis, nqs_model, k_max, n_particles, ps, st, beta)
    end
    println("熱平衡化が完了しました。")

    # === 4. 推定 ===
    # サンプリングとデータ収集
    # 現在の状態でのネットワーク出力を取得 [2, n_walkers]
    println("サンプリングを実施中...")
    acc_rate = 0f0
    all_states = CUDA.zeros(Int32, (2 * k_max + 1), 3, n_walkers * n_steps)
    all_outputs = CUDA.zeros(ComplexF32, n_walkers * n_steps)
    for step in 1:n_steps
        for _ in 1:n_interval
            # マルコフ連鎖を1ステップ進める
            n_accepted = CUDA.zeros(Float32, basis.n_walkers)
            sample_step!(sampler, basis, n_accepted, nqs_model, k_max, n_particles, ps, st, beta)
            acc_rate += sum(n_accepted) / basis.n_walkers
        end
        
        all_states[:, :, (step-1)*n_walkers+1 : step*n_walkers] .= basis.states
    end
    @printf(io, "accept ratio = %.4f\n", acc_rate / n_total)
    println("サンプリングが完了しました。")
   
    println("=== 推定を開始します ===")
    logw_lst = CUDA.zeros(Float32, n_total)
    for c in Iterators.partition(1:n_walkers, n_total)
        inputs_c = all_states[:, :, c]
        outputs_c = eval_complex_network(nqs_model, inputs_c, ps, st)
        all_outputs[c] .= outputs_c
        logw = beta .* real.(outputs_c)
        logw_lst[c] = logw
    end
    logw_lst .-= maximum(logw_lst)
    w_lst = exp.(logw_lst)
    w_sum = sum(w_lst)

    E_sum  = 0
    E2_sum = 0
    S2_sum = 0
    n1_sum = 0
    n2_sum = 0
    n3_sum = 0
    np0_sum = 0
    for c in Iterators.partition(1:n_walkers, n_total)
        inputs_c = all_states[:, :, c]
        outputs_c = all_outputs[c]

        E_loc_c = compute_local_energy(inputs_c, outputs_c, buffer.proposed_states, buffer.matrix_elements, params, basis.threads, nqs_model, ps, st)
        S2_loc_c = compute_local_S2(inputs_c, outputs_c, n_particles, k_max, basis.threads, nqs_model, ps, st)
      
        w = w_lst[c]
        E_sum  += sum(w .* E_loc_c)
        E2_sum += sum(w .* abs2.(E_loc_c))
        S2_sum += sum(w .* S2_loc_c)
        n1_sum += sum(transpose(w) .* inputs_c[:, 1, :])
        n2_sum += sum(transpose(w) .* inputs_c[:, 2, :])
        n3_sum += sum(transpose(w) .* inputs_c[:, 3, :])
        np0_sum += sum(transpose(w) .* inputs_c[k_max + 1, :, :])
    end
    E_mean  = sum(E_sum)  / w_sum
    E2_mean = sum(E2_sum) / w_sum
    S2_sum  = sum(S2_sum) / w_sum

    E_real = real(ComplexF32(E_mean))
    E_imag = imag(ComplexF32(E_mean))
    E_var  = Float32(E2_mean - abs2(E_mean))
    S2_real = real(ComplexF32(S2_sum))
    S2_imag = imag(ComplexF32(S2_sum))

    n1_mean = sum(n1_sum) / w_sum
    n2_mean = sum(n2_sum) / w_sum
    n3_mean = sum(n3_sum) / w_sum
    # l≠0 モードの総占有 = N - (l=0 の占有)。states[k_max+1, :, :] が l=0 の全スピン
    n_off = n_particles - sum(np0_sum) / w_sum

    E_real = real(E_mean)
    E_imag = imag(E_mean)
    E_var = Float32(E2_mean - abs2(E_mean))
    n1_mean = sum(all_states[:, 1, :]) / n_total
    n2_mean = sum(all_states[:, 2, :]) / n_total
    n3_mean = sum(all_states[:, 3, :]) / n_total
    @printf(io, "%10.8f, %10.8f, %10.8f, %10.8f, %10.8f, %6.5f, %6.5f, %6.5f, %6.8f, \n", 
    E_real, E_imag, E_var, S2_real, S2_imag, n1_mean, n2_mean, n3_mean, n_off)
 
    # 相関関数の評価
    inputs = Float32.(all_states[:, :, 1:n_walkers])
    outputs = eval_complex_network(nqs_model, inputs, ps, st)
    eval_space_correlation(all_states[:, :, 1:n_walkers], outputs, w_lst[1:n_walkers], k_max, basis.threads, n_walkers, nqs_model, ps, st, dirname)

    println("=== 計算が終了しました ===")
    close(io)
end

function eval_space_correlation(states, outputs, w, k_max, threads, n_walkers, nqs_model, ps, st, dirname)
    filename  = dirname * "space_correlation.txt"
    if isfile(filename)
        rm(filename)
    end
    touch(filename)
    
    ## 規格化定数
    w_sum = sum(w)

    ## 対角要素
    Ns_w = dropdims(sum(states, dims=1), dims=1)
    Ns_diag = Array(dropdims(sum(Ns_w .* (Ns_w .- Int32(1)) .* reshape(w, 1, :), dims=2), dims=2) ./ w_sum)
    n1_diag = Float32(Ns_diag[1])
    n2_diag = Float32(Ns_diag[2])
    n3_diag = Float32(Ns_diag[3])

    ## 運動量空間の相関関数
    rho2_q_loc = compute_local_correlation(states, outputs, k_max, threads, nqs_model, ps, st)
    rho2_q_mean = Array(dropdims(sum(rho2_q_loc .* reshape(w, 1, 1, :), dims=3), dims=3) ./ w_sum)   # [n_modes, 3]
    rho2_q_1 = rho2_q_mean[:, 1]
    rho2_q_2 = rho2_q_mean[:, 2]
    rho2_q_3 = rho2_q_mean[:, 3]
    rho2_q   = dropdims(sum(rho2_q_mean, dims=2), dims = 2)
    rho2_q_1[k_max + 1] = n1_diag
    rho2_q_2[k_max + 1] = n2_diag
    rho2_q_3[k_max + 1] = n3_diag
    rho2_q[k_max + 1]   = n1_diag + n2_diag + n3_diag

    # フーリエ変換
    L_box = Float32(2 * π)
    x_grid = Float32.(range(-L_box/2, L_box/2, length=1000))
    k_list = Float32.((2 * π / L_box) .* (-k_max:k_max))
    W = exp.(-1.0f0im .* x_grid .* k_list')
    cor1_x_vec = real.(W * rho2_q_1) ./ L_box
    cor2_x_vec = real.(W * rho2_q_2) ./ L_box
    cor3_x_vec = real.(W * rho2_q_3) ./ L_box
    cor_x_vec  = real.(W * rho2_q) ./ L_box

    @printf("n1_diag, n2_diag, n3_diag, max_correlation,\n") 
    @printf("%6.3f, %6.3f, %6.3f, %6.3f, \n", n1_diag, n2_diag, n3_diag, maximum(abs.(cor1_x_vec - cor3_x_vec)))

    open(filename, "a") do io
        @printf(io, "x, <n1>, <n2>, <n3>, <n_total>,\n")
    end
    for x in 1:1000
        cor1_x = cor1_x_vec[x]
        cor2_x = cor2_x_vec[x]
        cor3_x = cor3_x_vec[x]
        cor_x  = cor_x_vec[x]
        open(filename, "a") do io
            @printf(io, "%6.3f, %6.9f, %6.9f, %6.9f, %6.9f, \n", x_grid[x], cor1_x, cor2_x, cor3_x, cor_x)
        end
    end
    
    return nothing
end

# 実行
main()