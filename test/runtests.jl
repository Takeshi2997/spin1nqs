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
include("../src/hilbert.jl")
include("../src/model.jl")
include("../src/sampler.jl")
include("../src/physics.jl")
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
    open(filename, "a") do io
        @printf(io, "Epoch, Re<E>, Im<E>, <n1>, <n2>, <n3>,\n")
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
    n_walkers = train_config["n_walkers"]
    n_thermal = train_config["n_thermal"]
    n_steps = train_config["n_steps"]
    n_interval = train_config["n_interval"]
    n_epochs = train_config["n_epochs"]
    learning_rate = Float32(train_config["learning_rate"])
    epsilon = Float32(train_config["epsilon"])
    epsilon2 = Float32(train_config["epsilon2"])

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
    ps_cpu, st_cpu = load_nqs_model("./data/20260710_estimate/nqs_model_770_epoch20000.jld2")
    ps_cpu, st_cpu = expand_params(ps_cpu, 1, k_max, nqs_model)
   
    # 3モードのテスト配置 (例: 全粒子 l=0, sz=0)
    model3 = build_momentum_nqs(1, hidden_dim=hidden_dim)
    ps3, st3 = load_nqs_model("./data/20260710_estimate/nqs_model_770_epoch20000.jld2")
    model11 = build_momentum_nqs(5, hidden_dim=hidden_dim)
    ps11, st11 = expand_params(ps3, 1, 5, model11)

    occ3 = zeros(Int32, 3, 3);  occ3[2, 2] = 6
    occ11 = zeros(Int32, 11, 3); occ11[6, 2] = 6   # 同じ物理配置の11モード表現
    
    out3  = eval_complex_network(model3,  reshape(Float32.(occ3),  9,  1), ps3,  st3)
    out11 = eval_complex_network(model11, reshape(Float32.(occ11), 33, 1), ps11, st11)
    @assert isapprox(out3[1], out11[1]; atol=1e-5) "移植ミス: 列の対応がずれている"
    
    occ3 = zeros(Int32, 3, 3);  occ3[2, 1] = 3; occ3[2, 3] = 3
    occ11 = zeros(Int32, 11, 3); occ11[6, 1] = 3; occ11[6, 3] = 3 # 同じ物理配置の11モード表現
    
    out3  = eval_complex_network(model3,  reshape(Float32.(occ3),  9,  1), ps3,  st3)
    out11 = eval_complex_network(model11, reshape(Float32.(occ11), 33, 1), ps11, st11)
    @assert isapprox(out3[1], out11[1]; atol=1e-5) "移植ミス: 列の対応がずれている"

    occ3 = zeros(Int32, 3, 3);  occ3[2, 1] = 2; occ3[2, 2] = 2; occ3[2, 3] = 2
    occ11 = zeros(Int32, 11, 3); occ11[6, 1] = 2; occ11[6, 2] = 2; occ11[6, 3] = 2 # 同じ物理配置の11モード表現
    
    out3  = eval_complex_network(model3,  reshape(Float32.(occ3),  9,  1), ps3,  st3)
    out11 = eval_complex_network(model11, reshape(Float32.(occ11), 33, 1), ps11, st11)
    @assert isapprox(out3[1], out11[1]; atol=1e-5) "移植ミス: 列の対応がずれている"

    occ3 = zeros(Int32, 3, 3);  occ3[2, 1] = 1; occ3[2, 2] = 4; occ3[2, 3] = 1
    occ11 = zeros(Int32, 11, 3); occ11[6, 1] = 1; occ11[6, 2] = 4; occ11[6, 3] = 1 # 同じ物理配置の11モード表現
    
    out3  = eval_complex_network(model3,  reshape(Float32.(occ3),  9,  1), ps3,  st3)
    out11 = eval_complex_network(model11, reshape(Float32.(occ11), 33, 1), ps11, st11)
    @assert isapprox(out3[1], out11[1]; atol=1e-5) "移植ミス: 列の対応がずれている"

    exit()

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

    # === 4. メイン学習ループ ===
    for epoch in 1:n_epochs
        # このEpochでの観測量を蓄積するコンテナ
        E_loc_all = ComplexF32[]
        n1_all = Float32[]
        n2_all = Float32[]
        n3_all = Float32[]
        
        # A. サンプリングとデータ収集
        for step in 1:n_steps
            for _ in 1:n_interval
                # マルコフ連鎖を1ステップ進める
                sample_step!(sampler, basis, nqs_model, k_max, n_particles, ps, st)
            end
            
            # 現在の状態でのネットワーク出力を取得 [2, n_walkers]
            inputs = basis.states
            outputs = eval_complex_network(nqs_model, inputs, ps, st)

            # 局所エネルギー E_loc の計算
            E_loc = compute_local_energy(basis.states, outputs, buffer.proposed_states, buffer.matrix_elements, params, basis.threads, nqs_model, ps, st)

            E_mean = sum(E_loc) / n_walkers
            push!(E_loc_all, E_mean)

            n1_mean = sum(basis.states[:, 1, :]) / n_walkers
            n2_mean = sum(basis.states[:, 2, :]) / n_walkers
            n3_mean = sum(basis.states[:, 3, :]) / n_walkers
            n1_all = push!(n1_all, n1_mean)
            n2_all = push!(n2_all, n2_mean)
            n3_all = push!(n3_all, n3_mean)
        end
        
        # B. エネルギー期待値・粒子数期待値の算出
        E_mean = sum(E_loc_all) / n_steps
        E_real = real(E_mean)
        E_imag = imag(E_mean)
        n1_mean = sum(n1_all) / n_steps
        n2_mean = sum(n2_all) / n_steps
        n3_mean = sum(n3_all) / n_steps
        
        # C. パラメータ更新のための勾配計算と更新
        inputs = Float32.(basis.states)
        outputs = eval_complex_network(nqs_model, inputs, ps, st)
        E_loc_latest = compute_local_energy(basis.states, outputs, buffer.proposed_states, buffer.matrix_elements, params, basis.threads, nqs_model, ps, st)
        delta_p = compute_SR_update(nqs_model, ps, st, inputs, E_loc_latest, epsilon, epsilon2)

        # 相関関数の評価は ps 更新「前」に行う。
        # (basis.states は |psi_old|^2 からのサンプルであり、outputs も旧psでの評価なので、
        #  ps 更新後に呼ぶと compute_local_correlation 内部の psi(x') だけが新psになり、
        #  psi比 exp(log psi_new(x') - log psi_old(x)) が不整合になる)
        if epoch % save_iter == 0
            eval_space_correlation(basis.states, outputs, k_max, basis.threads, n_walkers, nqs_model, ps, st, dirname, epoch)
        end

        gnorm = sqrt(sum(abs2, delta_p))
        if gnorm > 1.0f0
            delta_p .*= 1.0f0 / gnorm
        end
        ps .= ps .- learning_rate .* delta_p

        # D. 進捗の表示
        if epoch % log_iter == 0 || epoch == 1
            @printf("Epoch %4d | <E> = %10.5f + i(%10.5f), <n1> = %6.3f, <n2> = %6.3f, <n3> = %6.3f\n", epoch, E_real, E_imag, n1_mean, n2_mean, n3_mean)
            open(filename, "a") do io
                @printf(io, "%4d, %10.5f, %10.5f, %6.3f, %6.3f, %6.3f,\n", epoch, E_real, E_imag, n1_mean, n2_mean, n3_mean)
            end
        end
        if epoch % save_iter == 0
            save_nqs_model(dirname, epoch, ps, st)
        end
    end
    
    println("=== 学習が正常に終了しました ===")
end

function eval_space_correlation(states, outputs, k_max, threads, n_walkers, nqs_model, ps, st, dirname, epoch)
    filename  = dirname * "/space_correlation_epoch$(epoch).txt"
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
    
    @printf("n1_diag, n2_diag, n3_diag, max_correlation\n") 
    @printf("%6.3f, %6.3f, %6.3f, %6.3f, \n", n1_diag, n2_diag, n3_diag, maximum(abs.(cor1_x_vec - cor3_x_vec)))

    return nothing 
end

# 実行
main()