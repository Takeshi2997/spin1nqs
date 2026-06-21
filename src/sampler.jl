module Sampler

using CUDA
using Random
using Lux
using ..Hilbert
using ..Model

export MCMCSampler, sample_step!

"""
MCMCの作業用メモリを管理する構造体
"""
struct MCMCSampler
    proposed_states::CuArray{Int32, 3}      # 提案状態を入れるバッファ
    current_inputs::CuArray{Float32, 3}     # NN入力用(現在)
    proposed_inputs::CuArray{Float32, 3}    # NN入力用(提案)
    rand_vals::CuArray{Float32, 1}          # 受容判定用の一様乱数バッファ

    function MCMCSampler(basis)
        # ウォーカー数分のメモリを初期化時に「一度だけ」確保する
        proposed_states = CUDA.zeros(Int32, size(basis.states))
        current_inputs = CUDA.zeros(Float32, size(basis.states))
        proposed_inputs = CUDA.zeros(Float32, size(basis.states))
        rand_vals = CUDA.zeros(Float32, basis.n_walkers)
        new(proposed_states, current_inputs, proposed_inputs, rand_vals)
    end
end

"""
全ウォーカーを並列に1ステップ進める関数
"""
function sample_step!(sampler::MCMCSampler, basis, model, kmax, n_particles, ps, st)
    
    # --- 1. 提案状態の生成 (Hilbert.jl) ---
    # basis.states に2体散乱を適用し、結果を sampler.proposed_states に書き込む
    Hilbert.generate_proposal!(basis.states, sampler.proposed_states, kmax, n_particles, basis.threads)
    
    # --- 2. 波動関数の評価 (Model.jl) ---
    # NNに入力するため Int32 -> Float32 へ型変換してバッファへコピー
    sampler.current_inputs .= Float32.(basis.states)
    sampler.proposed_inputs .= Float32.(sampler.proposed_states)

    # 現在の状態 x と提案状態 x' の log|Ψ| を計算 (Lux.apply)
    # 出力は [2, n_walkers] の行列
    log_psi_current, _ = Lux.apply(model, sampler.current_inputs, ps, st)
    log_psi_proposed, _ = Lux.apply(model, sampler.proposed_inputs, ps, st)
    println(log_psi_current[1, 1], log_psi_current[2, 1])

    # --- 3. メトロポリス判定の準備 ---
    rand!(sampler.rand_vals) # [0, 1) の乱数を生成
   
    # --- 4. 並列受容・棄却判定 ---
    blocks = ceil(Int, basis.n_walkers / basis.threads)
    @cuda threads=basis.threads blocks=blocks _accept_reject_kernel!(
        basis.states, sampler.proposed_states,
        log_psi_current, log_psi_proposed,
        sampler.rand_vals, basis.n_modes
    )
    basis.states = sampler.proposed_states

    return nothing
end

"""
各ウォーカーごとにメトロポリス判定を行い、受容なら状態を更新するカーネル
"""
function _accept_reject_kernel!(states, proposed_states, log_psi_cur, log_psi_prop, rand_vals, n_modes)
    # 自分が担当するウォーカー(列)のインデックス
    w = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if w <= size(states, 3)
        # 受容確率: P(x -> x') = |Ψ(x') / Ψ(x)|^2
        log_P = 2.0f0 * (log_psi_prop[1, w] - log_psi_cur[1, w])
        
        # log(乱数) < log(P) ならば受容
        if log(rand_vals[w]) < log_P
            # 受容: proposed_states の内容を states に上書きコピー
            for m in 1:n_modes
                for s in 1:3
                    states[m, s, w] = proposed_states[m, s, w]
                end
            end
        end
    end
    return nothing
end

end # module Sampler