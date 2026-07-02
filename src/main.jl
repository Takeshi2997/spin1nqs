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
include("optimise.jl")

using .Hilbert
using .Model
using .Sampler
using .Physics
using .Optimise

function main()
    dirname = "./data/" * Dates.format(now(), "yyyymmdd")
    filename  = dirname * "/data.txt"
    filename_nx  = dirname * "/distribution_n_particle.txt"
    if !isdir(dirname)
        mkdir(dirname)
    end
    if isfile(filename)
        rm(filename)
    end
    if isfile(filename_nx)
        rm(filename_nx)
    end
        
    touch(filename)
    touch(filename_nx)

    open(filename, "a") do io
        @printf(io, "Epoch, Re<E>, Im<E>, <n1>, <n2>, <n3>,\n")
    end
 
    # === 1. 物理・シミュレーションパラメータの設定 ===
    k_max = 5               # カットオフ波数 
    n_particles = 6         # 全粒子数 N
    target_Mz = 0           # 総磁化 M_z = 0 セクターに固定
    
    n_walkers = 1000         # 並列ウォーカー数 (バッチサイズ)
    n_thermal = 500         # 熱平衡化のための空回しステップ数
    n_steps = 100           # 1エポック（学習の1打）あたりのサンプリング数
    n_interval = 20         # サンプリングのインターバル
    n_epochs = 100        # 総学習Epoch数

    # ハミルトニアン係数（接触相互作用）
    params = SystemParams(
        k_max,
        2 * k_max + 1,
        1.0f0,  # hbar^2 / 2m
        0.0f0,  # c0 (密度相互作用)
        0.2f0   # c1 (スピン交換相互作用)
    )

    rng = Xoshiro(42)
    CUDA.allowscalar(false) # GPUのシリアルアクセス(低速化の原因)を禁止してデバッグ

    println("=== VMC駆動テストを起動します ===")
    println("環境: ", CUDA.functional() ? "GPU (CUDA)" : "CPU (警告: 動作が遅くなります)")

    # === 2. 各種構造体・ネットワークの初期化 ===
    # A. ヒルベルト空間の確保と初期状態の配置 (M_z=0に固定)
    basis = MomentumSpinorBasis(k_max, n_particles, 256, n_walkers)
    initialize_states!(basis, target_Mz)

    # B. 複素数出力NQSモデルの構築 (出力2ch)
    nqs_model = build_momentum_nqs(k_max, hidden_dim=128)
    ps_cpu, st_cpu = initialize_model(nqs_model, rng)
    
    # 重み(ps)と状態(st)をGPUへ転送
    ps = ComponentArray(ps_cpu) |> cu
    st = st_cpu |> cu

    # C. サンプラーバッファの確保
    sampler = MCMCSampler(basis)
    buffer = PhysicsBuffer(k_max, n_particles^2 * 2 * (2 * k_max + 1), n_walkers)

    # === 3. マルコフ連鎖の熱平衡化（Thermalization） ===
    println("マルコフ連鎖を熱平衡化中 ($(n_thermal) ステップ)...")
    for _ in 1:n_thermal
        sample_step!(sampler, basis, nqs_model, k_max, n_particles, ps, st)
        ## println(basis.states[:, :, 1])
    end
    println("熱平衡化が完了しました。")
    ## exit()

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
        delta_p = compute_SR_update(nqs_model, ps, st, inputs, E_loc_latest)
        learning_rate = 0.01f0
        ps .= ps .- learning_rate .* delta_p

        # D. 進捗の表示
        if epoch % 10 == 0 || epoch == 1
            @printf("Epoch %4d | <E> = %10.5f + i(%10.5f), <n1> = %6.3f, <n2> = %6.3f, <n3> = %6.3f\n", epoch, E_real, E_imag, n1_mean, n2_mean, n3_mean)
            open(filename, "a") do io
                @printf(io, "%4d, %10.5f, %10.5f, %6.3f, %6.3f, %6.3f,\n", epoch, E_real, E_imag, n1_mean, n2_mean, n3_mean)
            end
        end
    end

    # == 5. 粒子数分布の計算 ===
    sample_step!(sampler, basis, nqs_model, k_max, n_particles, ps, st)
    inputs = Float32.(basis.states)
    outputs = eval_complex_network(nqs_model, inputs, ps, st)
    rho_kq_loc = compute_local_density_matrix(basis.states, outputs, params, basis.threads, nqs_model, ps, st)
    rho_kq_mean = Array(dropdims(sum(rho_kq_loc, dims=2), dims=2) ./ n_walkers)
    rho1_kq = reshape(rho_kq_mean[:, 1], 2 * k_max + 1, 2 * k_max + 1)
    rho2_kq = reshape(rho_kq_mean[:, 2], 2 * k_max + 1, 2 * k_max + 1)
    rho3_kq = reshape(rho_kq_mean[:, 3], 2 * k_max + 1, 2 * k_max + 1)
    rho1_kq = vec((rho1_kq + rho1_kq') ./ 2)
    rho2_kq = vec((rho2_kq + rho2_kq') ./ 2)
    rho3_kq = vec((rho3_kq + rho3_kq') ./ 2)

    # フーリエ変換
    x_grid = Float32.(range(-5, 5, length=1000))
    k_list = Float32.((2 * π / 10) .* (-k_max:k_max))
    q_list = vec(k_list .- k_list')
    W = exp.(-1.0f0im .* x_grid .* q_list') ./ 10.0f0
    n1_x_vec = real.(W * rho1_kq)
    n2_x_vec = real.(W * rho1_kq)
    n3_x_vec = real.(W * rho1_kq)

    for x in 1:1000
        n1_x = n1_x_vec[x]
        n2_x = n2_x_vec[x]
        n3_x = n3_x_vec[x]
        open(filename_nx, "a") do io
            @printf(io, "%4d, %6.3f, %6.3f, %6.3f,\n", x, n1_x, n2_x, n3_x)
        end
    end

    # === 6. 学習済みモデルの保存 ===
    save_nqs_model(dirname, ps, st)
    
    println("=== 学習が正常に終了しました ===")
end

# 実行
main()