using CUDA
using Random
using Statistics
using Lux
using Optimisers
using Zygote
using Printf

# 自作モジュールの読み込み（同じディレクトリにあると仮定）
include("Hilbert.jl")
include("Model.jl")
include("Sampler.jl")
include("Physics.jl")

using .Hilbert
using .Model
using .Sampler
using .Physics

function main()
    # === 1. 物理・シミュレーションパラメータの設定 ===
    k_max = 5              # カットオフ波数 (モード総数: 2*5 + 1 = 11)
    n_particles = 10        # 全粒子数 N
    target_Mz = 0           # 総磁化 M_z = 0 セクターに固定
    
    n_walkers = 1000       # 並列ウォーカー数 (バッチサイズ)
    n_thermal = 500         # 熱平衡化のための空回しステップ数
    n_steps = 100           # 1エポック（学習の1打）あたりのサンプリング数
    n_epochs = 1          # 総学習Epoch数
    
    # ハミルトニアン係数（接触相互作用）
    params = SystemParams(
        k_max,
        2 * k_max + 1,
        0.5f0,  # hbar^2 / 2m
        1.0f0,  # c0 (密度相互作用)
        -0.2f0  # c2 (スピン交換相互作用)
    )

    rng = Xoshiro(42)
    CUDA.allowscalar(true) # GPUのシリアルアクセス(低速化の原因)を禁止してデバッグ

    println("=== VMC駆動テストを起動します ===")
    println("環境: ", CUDA.functional() ? "GPU (CUDA)" : "CPU (警告: 動作が遅くなります)")

    # === 2. 各種構造体・ネットワークの初期化 ===
    # A. ヒルベルト空間の確保と初期状態の配置 (M_z=0に固定)
    basis = MomentumSpinorBasis(k_max, n_particles, 256, n_walkers)
    initialize_states!(basis, target_Mz)

    # B. 複素数出力NQSモデルの構築 (出力2ch)
    nqs_model = build_momentum_nqs(k_max, hidden_dim=64)
    ps_cpu, st_cpu = initialize_model(nqs_model, rng)
    
    # 重み(ps)と状態(st)をGPUへ転送
    ps = ps_cpu |> cu
    st = st_cpu |> cu

    # C. サンプラーバッファの確保
    sampler = MCMCSampler(basis)
    buffer = PhysicsBuffer(k_max, n_particles^2 * 2 * k_max, n_walkers)

    # D. オプティマイザの準備 (ここでは簡易的に通常のAdamを採用)
    opt = Optimisers.Adam(0.001f0)
    opt_state = Optimisers.setup(opt, ps)

    # === 3. マルコフ連鎖の熱平衡化（Thermalization） ===
    println("マルコフ連鎖を熱平衡化中 ($(n_thermal) ステップ)...")
    for _ in 1:n_thermal
        sample_step!(sampler, basis, nqs_model, k_max, n_particles, ps, st)
        println(basis.states[:, :, 1])
    end
    println("熱平衡化が完了しました。")

    # === 4. メイン学習ループ ===
    for epoch in 1:n_epochs
        # このEpochでの観測量を蓄積するコンテナ
        E_loc_all = ComplexF32[]
        
        # (A) サンプリングとデータ収集
        for step in 1:n_steps
            # マルコフ連鎖を1ステップ進める
            sample_step!(sampler, basis, nqs_model, k_max, n_particles, ps, st)
            
            # 現在の状態でのネットワーク出力を取得 [2, n_walkers]
            inputs = Float32.(basis.states)
            outputs, _ = Lux.apply(nqs_model, inputs, ps, st)
            
            # 局所エネルギー E_loc の計算
            E_loc = compute_local_energy!(basis.states, outputs, buffer.proposed_states, buffer.matrix_elements, params, basis.threads, nqs_model, ps, st)
            
            # CPUへ集約して記録 (1ステップ分の全ウォーカーの平均)
            push!(E_loc_all, mean(Array(E_loc)))
        end
        
        # (B) エネルギー期待値の算出
        E_mean = mean(E_loc_all)
        E_real = real(E_mean)
        E_imag = imag(E_mean) # 統計が十分ならほぼ 0 になるはず
        
        # (C) VMC特有の「エネルギー勾配」の計算
        # VMCでは、通常の自動微分(loss = E)は使えません。
        # 損失関数を「2 * Re[ (E_loc - <E_loc>) * logΨ ]」の平均と定義すると、
        # その微分が、量子力学的なエネルギー勾配の数式と数学的に完全に一致します。
        
        # 1ステップ分の最新のサンプリング状態を代表として勾配を計算する例
        inputs = Float32.(basis.states)
        outputs, _ = Lux.apply(nqs_model, inputs, ps, st)
        E_loc_latest = compute_local_energy!(basis.states, outputs, buffer.proposed_states, buffer.matrix_elements, params, basis.threads, nqs_model, ps, st)
        E_loc_mean_scalar = mean(E_loc_latest) # このバッチの平均エネルギー
        
        # Zygoteを用いた自動微分
        # Luxのapplyがタプルを返すため、Lux.training_handleのような形で記述します
        grads = gradient(ps) do p
            # フォワードパス
            out, _ = Lux.apply(nqs_model, inputs, p, st)
            
            # 対数波動関数の実部と虚部
            @views log_psi_r = out[1, :]
            @views log_psi_i = out[2, :]
            
            # VMC勾配トリック用の重み (E_loc(x) - <E>) 
            # ※ p に関しては定数として扱うため、Zygoteの外で計算した E_loc_latest を使用
            ΔE_r = real.(E_loc_latest .- E_loc_mean_scalar)
            ΔE_i = imag.(E_loc_latest .- E_loc_mean_scalar)
            
            # トリック数式: 平均( 2 * (ΔE_r * log_psi_r + ΔE_i * log_psi_i) )
            loss = mean(2.0f0 .* (ΔE_r .* log_psi_r .+ ΔE_i .* log_psi_i))
            return loss
        end[1]

        # (D) パラメータの更新
        opt_state, ps = Optimisers.update(opt_state, ps, grads)

        # (E) 進捗の表示
        if epoch % 10 == 0 || epoch == 1
            @printf("Epoch %4d | <E> = %10.5f + i(%10.5f)\n", epoch, E_real, E_imag)
        end
    end
    println("=== 学習が正常に終了しました ===")
end

# 実行
main()