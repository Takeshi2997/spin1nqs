module Model

using Lux
using Zygote
using JLD2
using Functors
using Random
using NNlib 

export build_momentum_nqs, save_nqs_model, load_nqs_model, initialize_model, eval_complex_network, eval_complex_network_real, eval_complex_network_imag

"""
波数空間のスピン1ボゾン系向けNQSを構築する関数
"""
function build_momentum_nqs(k_max::Int; hidden_dim::Int=32)
    n_modes = 2 * k_max + 1
    input_features = n_modes * 3 # (波数モード数) × (スピン3成分)
    
    # Lux.Chain でネットワークを定義
    model = Chain(
        # 1. テンソルの平坦化
        # 入力 [n_modes, 3, n_walkers] を [input_features, n_walkers] に変換
        FlattenLayer(), 

        # 2. 全結合層
        Dense(input_features => hidden_dim, tanh),
        ## Dense(hidden_dim => hidden_dim, tanh),

        # 3. 出力層
        # 各ウォーカーに対して対数振幅 logΨの実部と虚部を出力
        Dense(hidden_dim => 2) 
    )
    
    return model
end

"""
保存処理 学習完了後やチェックポイント
"""
function save_nqs_model(dirname, epoch, ps_gpu, st_gpu)
    # 1. GPU (CuArray) から CPU (標準のArray) へ変換
    ps_cpu = fmap(Array, ps_gpu)
    st_cpu = fmap(Array, st_gpu)
    
    # 2. JLD2でファイルに保存
    num_params = Lux.parameterlength(ps_cpu)
    filename = dirname * "/nqs_model__$(num_params)_epoch$(epoch).jld2"
    @save filename ps_cpu st_cpu
    println("モデルを $(filename) に保存しました。")
end

"""
読み込み処理 計算の再開やデータ解析時
"""
function load_nqs_model(filename)
    # 1. JLD2ファイルからCPUメモリへ読み込み
    @load filename ps_cpu st_cpu
    
    # 2. GPUへ転送
    ps_gpu = fmap(x -> CuArray{Float32}(x), ps_cpu)
    st_gpu = fmap(x -> CuArray{Float32}(x), st_cpu)
    
    println("モデルを $(filename) から読み込み、GPUに転送しました。")
    return ps_gpu, st_gpu
end

"""
モデルの初期化と、GPU(CUDA)への転送準備を行うヘルパー関数
"""
function initialize_model(model, rng::AbstractRNG)
    # Luxでは、パラメータ(ps)と状態(st)を明示的に初期化します
    ps, st = Lux.setup(rng, model)
    return ps, st
end

"""
与えられた入力に対して、複素数の波動関数を評価する関数
"""
function eval_complex_network(model, inputs, ps, st)
    # 入力は [n_modes, 3, n_walkers] の形状を想定
    # Lux.apply は [2, n_walkers] の出力を返す（1行目: log|Ψ|の実部、2行目: log|Ψ|の虚部）
    outputs, _ = Lux.apply(model, Float32.(inputs), ps, st)
    
    # 複素数の波動関数 Ψ を構築
    log_psi_real = outputs[1, :]
    log_psi_imag = outputs[2, :]

    return log_psi_real .+ im .* log_psi_imag
end

function eval_complex_network_real(model, inputs, ps, st)
    return real.(eval_complex_network(model, inputs, ps, st))
end

function eval_complex_network_imag(model, inputs, ps, st)
    return imag.(eval_complex_network(model, inputs, ps, st))
end

end # module Model