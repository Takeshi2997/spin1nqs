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
    n_thermal = estimate_config["n_thermal"]
    n_interval = estimate_config["n_interval"]
    n_steps = estimate_config["n_steps"]
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
    srcdir = "./data/20260725/"
    epoch = 7000
    filename = "nqs_model_4610_epoch" * string(epoch) * ".jld2"
    cp(srcdir * filename, dirname * filename, force=true)
    ps_cpu, st_cpu = load_nqs_model(dirname * filename)

    filename  = dirname * "/data_log_epoch" * string(epoch) * ".txt"
    io = open(filename, "w")
   

##     # 重み(ps)と状態(st)をGPUへ転送
##     ps = ComponentArray(ps_cpu) |> cu
##     st = st_cpu |> cu

##     # C. サンプラーバッファの確保
##     sampler = MCMCSampler(basis)
##     buffer = PhysicsBuffer(k_max, min(n_particles, 3 * (2 * k_max + 1))^2 * 3 * (2 * k_max + 1), n_walkers)
## 
##     # === 3. マルコフ連鎖の熱平衡化（Thermalization） ===
##     acc_rate = 0f0
##     println("マルコフ連鎖を熱平衡化中 ($(n_thermal) ステップ)...")
##     for step in 1:n_thermal
##         accepted = CUDA.zeros(Float32, basis.n_walkers)
##         sample_step!(sampler, basis, accepted, nqs_model, k_max, n_particles, ps, st)
##         acc_rate += sum(accepted) / basis.n_walkers
##     end
##     @printf(io, "accept ratio = %.4f\n", acc_rate / n_thermal)
##     println("熱平衡化が完了しました。")

##     # === 4. 推定 ===
##     # サンプリングとデータ収集
##     # 現在の状態でのネットワーク出力を取得 [2, n_walkers]
##     all_states = CUDA.zeros(Int32, (2 * k_max + 1), 3, n_walkers * n_steps)
##     for step in 1:n_steps
##         for _ in 1:n_interval
##             # マルコフ連鎖を1ステップ進める
##             sample_step!(sampler, basis, nqs_model, k_max, n_particles, ps, st)
##         end
##         
##         all_states[:, :, (step-1)*n_walkers+1 : step*n_walkers] .= basis.states
##     end

    E_loc = ComplexF32[]
    E2_loc = Float32[]
    S2_loc = ComplexF32[]
##     for c in Iterators.partition(1:n_total, n_walkers)
##         inputs_c = all_states[:, :, c]
##         outputs_c = eval_complex_network(nqs_model, inputs_c, ps, st)
##         E_loc_c = compute_local_energy(inputs_c, outputs_c, buffer.proposed_states, buffer.matrix_elements, params, basis.threads, nqs_model, ps, st)
##         S2_loc_c = compute_local_S2(inputs_c, outputs_c, n_particles, k_max, basis.threads, nqs_model, ps, st)
## 
##         append!(E_loc, E_loc_c)
##         append!(E2_loc, abs2.(E_loc_c))
##         append!(S2_loc, S2_loc_c)
##     end

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

    exit()

    @printf(io, "<S²> = %.4f + i(%.4f)\n", real(sum(S2_loc)) / n_total, imag(sum(S2_loc)) / n_total)
    E_loc_real = Array(real.(E_loc))
    println(io, "min = ", minimum(E_loc_real))
    println(io, "max = ", maximum(E_loc_real))
    println(io, "median = ", median(E_loc_real))
    # 上位10個
    println(io, sort(E_loc_real, rev=true)[1:10])

##     states_host = Array(all_states)   # [n_modes, 3, n_walkers]
##     configs = [vec(states_host[:,:,w]) for w in 1:n_total]
##     n_unique = length(unique(configs))
##     println(io, "ユニーク配置数: $n_unique / $n_walkers")
    
##     # 最頻配置の占有率
##     cm = countmap(configs)
##     top = sort(collect(cm), by=x->-x[2])[1:5]
##     for (cfg, cnt) in top
##         println(io, "  $cnt 匹 (", round(100*cnt/n_walkers, digits=1), "%)")
##     end

##     # 状態の詳細
##     configs = [copy(states_host[:,:,w]) for w in 1:n_walkers]
##     cm = countmap(configs)
##     for (cfg, cnt) in sort(collect(cm), by=x->-x[2])[1:6]
##         println(io, "$(cnt) 匹 ($(round(100cnt/n_walkers,digits=1))%)")
##         for s in 1:3
##             occ = [cfg[m,s] for m in 1:11]
##             println(io, "  sz=$(s-2): ", occ, "  (N_s = $(sum(occ)))")
##         end
##         # 全運動量とエネルギー的な特徴
##         P = sum(cfg[m,s]*(m-6) for m in 1:11, s in 1:3)
##         Ekin = sum(cfg[m,s]*(m-6)^2 for m in 1:11, s in 1:3)
##         println(io, "  P = $P, E_kin = $Ekin")
##     end

    # エネルギー期待値・粒子数期待値の算出
    E_mean = sum(E_loc) / n_total
    E_real = real(E_mean)
    E_imag = imag(E_mean)
    E2_mean = sum(E2_loc) / n_total
    VarE = Float32(E2_mean - abs2(E_mean))
    n1_mean = sum(all_states[:, 1, :]) / n_total
    n2_mean = sum(all_states[:, 2, :]) / n_total
    n3_mean = sum(all_states[:, 3, :]) / n_total
    @printf(io, "E_real, E_imag, VarE, n1_mean, n2_mean, n3_mean, \n")
    @printf(io, "%10.5f, %10.5f, %10.5f, %10.5f, %6.3f, %6.3f,\n", E_real, E_imag, VarE, n1_mean, n2_mean, n3_mean)

##     # 相関関数の評価
##     inputs = Float32.(all_states[:, :, 1:n_walkers])
##     outputs = eval_complex_network(nqs_model, inputs, ps, st)
##     eval_space_correlation(all_states[:, :, 1:n_walkers], outputs, k_max, basis.threads, n_walkers, nqs_model, ps, st, dirname)

    println("=== 計算が終了しました ===")
    close(io)
end

function eval_space_correlation(states, outputs, k_max, threads, n_walkers, nqs_model, ps, st, dirname)
    filename  = dirname * "space_correlation.txt"
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