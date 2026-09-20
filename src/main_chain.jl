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
using ArgParse
using Functors
using JLD2

include("model.jl")
include("optimise.jl")
include("hilbert.jl")
include("sampler.jl")
include("physics.jl")

using CUDA
using .Model
using .Optimise
using .Hilbert
using .Sampler
using .Physics

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

function create_unique_dir(base_name::String)
    # 1. 元の名前のディレクトリが存在しない場合はそのまま作成
    if !isdir(base_name)
        mkpath(base_name)
        return base_name
    end

    # 2. すでに存在する場合は枝番（_1, _2...）を付けて空きを探す
    i = 1
    while true
        new_name = "$(base_name)_$(i)"
        if !isdir(new_name)
            mkpath(new_name)
            return new_name
        end
        i += 1
    end
end

function parse_commandline()
    # 設定オブジェクトを作成
    s = ArgParseSettings()

    # 受け付ける引数の定義
    @add_arg_table! s begin
        "--params"
            help = "パラメータファイル名"
            arg_type = String
            default = "config_local.toml"
        "--k_max"
            help = "カットオフモード数"
            arg_type = Int
            default = 5
        "--n"
            help = "粒子数"
            arg_type = Int
            default = 8
        "--c1"
            help = "相互作用係数"
            arg_type = Float32
            default = 2.0
        "--n_epoch"
            help = "エポック数"
            arg_type = Int
            default = 10000
        "--init"
            help = "初期化ファイル名"
            arg_type = String
            default = ""
        "--out"
            help = "出力ディレクトリ名"
            arg_type = String
            default = "none"
    end

    # 実際のコマンドライン引数(ARGS)を解析して辞書(Dict)で返す
    return parse_args(s)
end

function main()
    dirname = "./data/" * "n_scan_c1_const" ## Dates.format(now(), "yyyymmdd")
    dirname = create_unique_dir(dirname)
    filename  = dirname * "/data.txt"

    if isfile(filename)
        rm(filename)
    end
    touch(filename)
    commit = try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end
    open(filename, "a") do io
        @printf(io, "[%s] ==== 学習開始 （Git commit:%s）====\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), commit)
        @printf(io, "Epoch, UnixTime, Re<E>, Im<E>, VarE, Re<S2>, Im<S2>, <n1>, <n2>, <n3>, n_off, n_eff, n_clipping,\n")
    end
 
    # === 1. 物理・シミュレーションパラメータの設定 ===
    args = parse_commandline()

    config_path = args["params"]
    println("🔧 Loading configuration from: ", config_path)
     
    # 2. TOMLファイルのパース
    config = TOML.parsefile(config_path)

    # システム設定の読み込み
    sys_config = config["system"]
    k_max = args["k_max"]
    config["system"]["k_max"] = k_max
    n_particles = args["n"]
    config["system"]["n_particles"] = n_particles
    hbar2_over_2m = Float32(sys_config["hbar2_over_2m"])
    c0 = Float32(sys_config["c0"])
    c1 = args["c1"]
    config["system"]["c1"] = c1
    target_Mz = sys_config["target_Mz"]

    # 学習設定の読み込み
    train_config = config["training"]
    chunk = train_config["chunk"]
    n_walkers = train_config["n_walkers"]
    n_thermal = train_config["n_thermal"]
    n_steps = train_config["n_steps"]
    n_interval = train_config["n_interval"]
    n_epochs = args["n_epoch"]
    config["training"]["n_epochs"] = n_epochs
    learning_rate = Float32(train_config["learning_rate"])
    epsilon = Float32(train_config["epsilon"])
    epsilon2 = Float32(train_config["epsilon2"])
    decay = Float32(train_config["decay"])
    lambda_min = Float32(train_config["lambda_min"])
    clipping_threshold = Float32(train_config["clipping_threshold"])
    beta = Float32(train_config["beta"])
    p_spin = Float32(train_config["p_spin"])
    n_total = n_walkers * n_steps

    # モデル設定の読み込み
    model_config = config["model"]
    hidden_dim = model_config["hidden_dim"]

    # IO設定の読み込み
    io_config = config["io"]
    log_iter = io_config["log_iter"]
    save_iter = io_config["save_iter"]

    open(dirname * "/config.toml", "w") do io
        TOML.print(io, config, sorted=true)
    end
   
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
    initfilename = args["init"]
    if initfilename == "fresh" || isfile(initfilename) == false
        ps_cpu, st_cpu = initialize_model(nqs_model, rng)
    else
        ps_cpu, st_cpu = load_nqs_model(initfilename)
        rm(initfilename)
    end
    e_start = 1
    n_params = Lux.parameterlength(ps_cpu)

    # 重み(ps)と状態(st)をGPUへ転送
    ps = ComponentArray(ps_cpu) |> cu
    st = st_cpu |> cu

    ## rule = Optimisers.Adam()
    ## opt_state = Optimisers.setup(rule, ps)

    # C. サンプラーバッファの確保
    sampler = MCMCSampler(basis)
    buffer = PhysicsBuffer(k_max, min(n_particles, 3 * (2 * k_max + 1))^2 * 3 * (4 * k_max + 1), chunk)
    all_states = CUDA.zeros(Int32, (2 * k_max + 1), 3, n_walkers * n_steps)
    all_outputs = CUDA.zeros(ComplexF32, n_walkers * n_steps)
    O_sum   = CUDA.zeros(ComplexF32, n_params)
    OE_sum  = CUDA.zeros(ComplexF32, n_params)
    OO_sum  = CUDA.zeros(ComplexF32, n_params, n_params)
 
    # === 3. マルコフ連鎖の熱平衡化（Thermalization） ===
    println("マルコフ連鎖を熱平衡化中 ($(n_thermal) ステップ)...")
    for step in 1:n_thermal
        sample_step!(sampler, basis, nqs_model, k_max, n_particles, ps, st, beta, p_spin)
    end
    println("熱平衡化が完了しました。")

    ## report("Initialize")
    # === 4. メイン学習ループ ===
    n_clipping = 0
    for epoch in e_start:n_epochs
        ## println("########## Epoch Start! ##########")
        ## prof = CUDA.@profile begin
        # A. サンプリングとデータ収集
        for step in 1:n_steps
            for _ in 1:n_interval
                # マルコフ連鎖を1ステップ進める
                sample_step!(sampler, basis, nqs_model, k_max, n_particles, ps, st, beta, p_spin)
            end
            
            all_states[:, :, (step-1)*n_walkers+1 : step*n_walkers] .= basis.states
        end
        ## report("Sampling")

        # B. エネルギー期待値・粒子数期待値の算出
        logw_lst = CUDA.zeros(Float32, n_total)
        for c in Iterators.partition(1:n_total, chunk)
            inputs_c = all_states[:, :, c]
            outputs_c = eval_complex_network(nqs_model, inputs_c, ps, st)
            all_outputs[c] .= outputs_c
            logw = beta .* real.(outputs_c)
            logw_lst[c] = logw
        end
        logw_lst .-= maximum(logw_lst)
        w_lst = exp.(logw_lst)
        w_sum = sum(w_lst)
        w2 = w_sum^2 / sum(w_lst.^2) / n_total

        E_sum  = 0
        E2_sum = 0
        S2_sum = 0
        n1_sum = 0
        n2_sum = 0
        n3_sum = 0
        np0_sum = 0
        O_sum   = CUDA.zeros(ComplexF32, n_params)
        OE_sum  = CUDA.zeros(ComplexF32, n_params)
        OO_sum  = CUDA.zeros(ComplexF32, n_params, n_params)
        for c in Iterators.partition(1:n_total, chunk)
            inputs_c = all_states[:, :, c]
            outputs_c = all_outputs[c]

            ## report("Evaluate network")
            E_loc_c = compute_local_energy(inputs_c, outputs_c, buffer.proposed_states, buffer.matrix_elements, params, basis.threads, nqs_model, ps, st)
            S2_loc_c = compute_local_S2(inputs_c, outputs_c, n_particles, k_max, basis.threads, nqs_model, ps, st)
            ## report("Compute local Energy")
            ## O_c = compute_jacobian(nqs_model, inputs_c, ps, st)  # [n_params, length(c)]
            inputs_tmp = Float32.(reshape(inputs_c, (2 * k_max + 1) * 3, :))
            O_c, _ = compute_O_bar(inputs_tmp, ps)
            ## report("Compute jacobian")
          
            w = w_lst[c]
            E_sum  += sum(w .* E_loc_c)
            E2_sum += sum(w .* abs2.(E_loc_c))
            S2_sum += sum(w .* S2_loc_c)
            n1_sum += sum(transpose(w) .* inputs_c[:, 1, :])
            n2_sum += sum(transpose(w) .* inputs_c[:, 2, :])
            n3_sum += sum(transpose(w) .* inputs_c[:, 3, :])
            np0_sum += sum(transpose(w) .* inputs_c[k_max + 1, :, :])
            O_sum  .+= dropdims(sum(transpose(w) .* O_c, dims=2), dims=2)
            OE_sum .+= O_c * (w .* E_loc_c)
            OO_sum .+= (transpose(w) .* conj.(O_c)) * transpose(O_c)
        end
        E_mean  = sum(E_sum)  / w_sum
        E2_mean = sum(E2_sum) / w_sum
        S2_sum  = sum(S2_sum) / w_sum
        O_mean  = O_sum  ./ w_sum
        OE_mean = OE_sum ./ w_sum
        OO_mean = OO_sum ./ w_sum

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

        # C. 進捗の表示
        if epoch % log_iter == 0
            n_clipping = 0
        end
        if epoch % log_iter == 0 || epoch == e_start
            @printf("[%s] Epoch %4d | <E> = %10.5f, Var = %6.5f, <S2> = %10.5f, <n1> = %6.3f, <n2> = %6.3f, <n3> = %6.3f, n_off = %6.3f, n_clipping = %4d,\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), 
            epoch, E_real, E_var, S2_real, n1_mean, n2_mean, n3_mean, n_off, n_clipping)
            open(filename, "a") do io
                @printf(io, "%4d, %.3f, %10.8f, %10.8f, %10.8f, %10.8f, %10.8f, %6.5f, %6.5f, %6.5f, %6.8f, %6.8f, %4d,\n", 
                epoch, time(), E_real, E_imag, E_var, S2_real, S2_imag, n1_mean, n2_mean, n3_mean, n_off, w2, n_clipping)
            end
        end
        if epoch % save_iter == 0
            save_nqs_model(dirname, epoch, ps, st)
        end

        ## report("Compute Average")

        # D. パラメータ更新のための勾配計算と更新
        ## delta_p = compute_SR_update(nqs_model, ps, st, inputs, E_loc, epoch, epsilon, epsilon2)

        ## SR法
        delta_p = SR_update(O_mean, OO_mean, OE_mean, E_mean, epoch, epsilon, epsilon2, decay, lambda_min)
        ## report("SR")

        gnorm = sqrt(sum(abs2, delta_p))
        if gnorm > clipping_threshold
            delta_p .*= 1.0f0 / gnorm
            n_clipping += 1
        end
        ps .= ps .- learning_rate .* delta_p

        ## report("End")
        ## end
        ## display(prof)
    end
    
    ps_cpu = fmap(Array, ps)
    st_cpu = fmap(Array, st)
    filename = args["out"]
    if filename !== "none"
        @save filename ps_cpu st_cpu
    end

    println("=== 学習が正常に終了しました ===")
end

# 実行
main()
