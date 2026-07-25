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
    ps_cpu, st_cpu = load_nqs_model("./data/20260724_estimated/nqs_model_4610_epoch3000.jld2")
    e_start = 3001
    n_params = Lux.parameterlength(ps_cpu)

    # 重み(ps)と状態(st)をGPUへ転送
    ps = ComponentArray(ps_cpu) |> cu
    st = st_cpu |> cu

    rule = Optimisers.Adam()
    opt_state = Optimisers.setup(rule, ps)

    # C. サンプラーバッファの確保
    sampler = MCMCSampler(basis)
    buffer = PhysicsBuffer(k_max, min(n_particles, 3 * (2 * k_max + 1))^2 * 3 * (4 * k_max + 1), chunk)

    # === 3. マルコフ連鎖の熱平衡化（Thermalization） ===
    println("マルコフ連鎖を熱平衡化中 ($(n_thermal) ステップ)...")
    for step in 1:n_thermal
        sample_step!(sampler, basis, nqs_model, k_max, n_particles, ps, st)
    end
    println("熱平衡化が完了しました。")

    ## report("Initialize")
    # === 4. メイン学習ループ ===
    all_states = CUDA.zeros(Int32, (2 * k_max + 1), 3, n_walkers * n_steps)
    n_clipping = 0
    for epoch in e_start:n_epochs
        ## println("########## Epoch Start! ##########")
        ## prof = CUDA.@profile begin
        # A. サンプリングとデータ収集
        for step in 1:n_steps
            for _ in 1:n_interval
                # マルコフ連鎖を1ステップ進める
                sample_step!(sampler, basis, nqs_model, k_max, n_particles, ps, st)
            end
            
            all_states[:, :, (step-1)*n_walkers+1 : step*n_walkers] .= basis.states
        end
        ## report("Sampling")

        # B. エネルギー期待値・粒子数期待値の算出
        E_sum  = 0.0 + 0.0im
        S2_sum = 0.0 + 0.0im
        E2_sum  = 0.0
        O_sum   = CUDA.zeros(ComplexF32, n_params)           # Σ_x O_k(x)
        OO_sum  = CUDA.zeros(ComplexF32, n_params, n_params) # Σ_x O_k*(x) O_l(x)
        OE_sum  = CUDA.zeros(ComplexF32, n_params)           # Σ_x O_k*(x) E_loc(x)
        for c in Iterators.partition(1:n_total, chunk)
            inputs_c = all_states[:, :, c]                    # このチャンクだけGPUで処理
            outputs_c = eval_complex_network(nqs_model, inputs_c, ps, st)
            ## report("Evaluate network")
            E_loc_c = compute_local_energy(inputs_c, outputs_c, buffer.proposed_states, buffer.matrix_elements, params, basis.threads, nqs_model, ps, st)
            S2_loc_c = compute_local_S2(inputs_c, outputs_c, n_particles, k_max, basis.threads, nqs_model, ps, st)
            ## report("Compute local Energy")
            ## O_c = compute_jacobian(nqs_model, inputs_c, ps, st)  # [n_params, length(c)]
            inputs_tmp = Float32.(reshape(inputs_c, (2 * k_max + 1) * 3, :))
            O_c, _ = compute_O_bar(inputs_tmp, ps)
            ## report("Compute jacobian")

            E_sum  += sum(E_loc_c)
            E2_sum += sum(abs2.(E_loc_c))
            O_sum .+= dropdims(sum(O_c, dims=2), dims=2)
            OE_sum .+= O_c * E_loc_c
            OO_sum .+= conj.(O_c) * transpose(O_c)
            S2_sum += sum(S2_loc_c)
        end
        E_mean = E_sum / n_total
        E2_mean = E2_sum / n_total
        O_mean = O_sum ./ n_total
        OO_mean = OO_sum ./ n_total
        OE_mean = OE_sum ./ n_total
        S2_sum = S2_sum / n_total

        ## inputs = Float32.(all_states)
        ## outputs = eval_complex_network(nqs_model, inputs, ps, st)
        ## E_loc = compute_local_energy(all_states, outputs, buffer.proposed_states, buffer.matrix_elements, params, basis.threads, nqs_model, ps, st)
        ## compare_SR(nqs_model, ps, st, all_states, E_loc, O_mean, OO_mean, OE_mean, E_mean)
        ## E_mean = ComplexF64(sum(E_loc)) / n_samples
        ## E2_sum = Float64(sum(abs2.(E_loc)))
        E_real = real(ComplexF32(E_mean))
        E_imag = imag(ComplexF32(E_mean))
        E_var  = Float32(E2_mean - abs2(E_mean))
        S2_real = real(ComplexF32(S2_sum))
        S2_imag = imag(ComplexF32(S2_sum))

        n1_mean = sum(all_states[:, 1, :]) / n_total
        n2_mean = sum(all_states[:, 2, :]) / n_total
        n3_mean = sum(all_states[:, 3, :]) / n_total
        # l≠0 モードの総占有 = N - (l=0 の占有)。states[k_max+1, :, :] が l=0 の全スピン
        n_off = Float64(n_particles) - Float64(sum(all_states[k_max + 1, :, :])) / n_total

        # 相関関数の評価は ps 更新「前」に行う。
        # (basis.states は |psi_old|^2 からのサンプルであり、outputs も旧psでの評価なので、
        #  ps 更新後に呼ぶと compute_local_correlation 内部の psi(x') だけが新psになり、
        #  psi比 exp(log psi_new(x') - log psi_old(x)) が不整合になる)
        # C. 進捗の表示
        if epoch % log_iter == 0 || epoch == e_start
            @printf("[%s] Epoch %4d | <E> = %10.5f + i(%10.5f), Var = %6.5f, <S2> = %10.5f + i(%10.5f), <n1> = %6.3f, <n2> = %6.3f, <n3> = %6.3f, n_off = %6.3f, n_clipping = %4d,\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), 
            epoch, E_real, E_imag, E_var, S2_real, S2_imag, n1_mean, n2_mean, n3_mean, n_off, n_clipping)
            open(filename, "a") do io
                @printf(io, "%4d, %.3f, %10.8f, %10.8f, %10.8f, %10.8f, %10.8f, %6.5f, %6.5f, %6.5f, %6.8f, %4d,\n", 
                epoch, time(), E_real, E_imag, E_var, S2_real, S2_imag, n1_mean, n2_mean, n3_mean, n_off, n_clipping)
            end

            n_clipping = 0
        end
        if epoch % save_iter == 0
            inputs = Float32.(all_states[:, :, 1:chunk])
            outputs = eval_complex_network(nqs_model, inputs, ps, st)
            eval_space_correlation(all_states[:, :, 1:chunk], outputs, k_max, basis.threads, n_total, nqs_model, ps, st, dirname, epoch)
            save_nqs_model(dirname, epoch, ps, st)
        end

        ## report("Compute Average")

        # D. パラメータ更新のための勾配計算と更新
        ## delta_p = compute_SR_update(nqs_model, ps, st, inputs, E_loc, epoch, epsilon, epsilon2)

        ## SR法
        ## delta_p = SR_update(O_mean, OO_mean, OE_mean, E_mean, epoch, epsilon, epsilon2, decay, lambda_min)
        ## ## report("SR")

        ## gnorm = sqrt(sum(abs2, delta_p))
        ## if gnorm > clipping_threshold
        ##     delta_p .*= 1.0f0 / gnorm
        ##     n_clipping += 1
        ## end
        ## ps .= ps .- learning_rate .* delta_p

        R = real.(OE_mean .- O_mean .* E_mean)
        opt_state, ps = Optimisers.update!(opt_state, ps, R)

        ## report("End")
        ## end
        ## display(prof)
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
    
    @printf("n1_diag, n2_diag, n3_diag, max_correlation,\n") 
    @printf("%6.3f, %6.3f, %6.3f, %6.3f, \n", n1_diag, n2_diag, n3_diag, maximum(abs.(cor1_x_vec - cor3_x_vec)))

    return nothing 
end

# 実行
main()
