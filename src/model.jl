module Model

using Lux
using Zygote
using Random
using NNlib # 活性化関数（tanh, geluなど）を使用するために必要

export build_momentum_nqs, initialize_model, eval_complex_network, eval_complex_network_real, eval_complex_network_imag

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

        # 2. 全結合層（非局所的な波数間の相関を学習）
        # 活性化関数には、最適化が安定しやすい滑らかな関数（tanhなど）を採用
        Dense(input_features => hidden_dim, tanh),
        Dense(hidden_dim => hidden_dim, tanh),

        # 3. 出力層
        # 各ウォーカーに対して対数振幅 logΨの実部と虚部を出力
        Dense(hidden_dim => 2) 
    )
    
    return model
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

    phases = Zygote.@ignore begin
        return _compute_phase_correction(inputs)
    end

    return log_psi_real .+ im .* log_psi_imag .+ im .* phases
end

function eval_complex_network_real(model, inputs, ps, st)
    # 入力は [n_modes, 3, n_walkers] の形状を想定
    # Lux.apply は [2, n_walkers] の出力を返す（1行目: log|Ψ|の実部、2行目: log|Ψ|の虚部）
    outputs, _ = Lux.apply(model, Float32.(inputs), ps, st)
    
    # 複素数の波動関数 Ψ を構築
    log_psi_real = outputs[1, :]

    return log_psi_real
end

function eval_complex_network_imag(model, inputs, ps, st)
    # 入力は [n_modes, 3, n_walkers] の形状を想定
    # Lux.apply は [2, n_walkers] の出力を返す（1行目: log|Ψ|の実部、2行目: log|Ψ|の虚部）
    outputs, _ = Lux.apply(model, Float32.(inputs), ps, st)
    
    # 複素数の波動関数 Ψ を構築
    log_psi_imag = outputs[2, :]

    phases = Zygote.@ignore begin
        return _compute_phase_correction(inputs)
    end

    return log_psi_imag .+ phases
end

function _compute_phase_correction(inputs)
    n0 = dropdims(sum(inputs[:, 2, :], dims=1), dims=1)
    signs = (-1.0f0) .^ (Float32.(n0) ./ 2.0f0)
    return (signs .< 0) .* Float32(π)
end

end # module Model